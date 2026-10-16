package main

import (
	"strings"
	"testing"
)

func TestBuildToolInstructionsUsesFlatCallForm(t *testing.T) {
	instructions := buildToolInstructions(nil)
	if want := `[TOOL_CALL]{"name":"TOOL_NAME","arguments":{}}[/TOOL_CALL]`; !strings.Contains(instructions, want) {
		t.Fatalf("instructions do not contain flat call form %q", want)
	}
	if want := `CRITICAL: When a tool is needed, you MUST output the`; !strings.Contains(instructions, want) {
		t.Fatalf("instructions do not contain critical guidance for reasoning models: %q", want)
	}
	if strings.Contains(instructions, `"function":{"name"`) {
		t.Fatal("instructions still advertise the nested call form")
	}
}

func TestRkToolParserKeepsSplitOpeningTagPrivate(t *testing.T) {
	p := NewRkToolParser()
	visible, calls := p.ProcessChunk("I will inspect it. [TOOL_")
	if visible != "I will inspect it. " || len(calls) != 0 {
		t.Fatalf("first chunk = (%q, %d calls), want visible text only", visible, len(calls))
	}

	visible, calls = p.ProcessChunk("CALL]{\"function\":{\"name\":\"get_devices\",\"arguments\":{}}}[/TOOL_CALL]Done.")
	if visible != "Done." || len(calls) != 1 {
		t.Fatalf("second chunk = (%q, %d calls), want trailing text and one call", visible, len(calls))
	}
	if calls[0].Function.Name != "get_devices" || string(calls[0].Function.Arguments) != "{}" {
		t.Fatalf("unexpected call: %#v", calls[0])
	}
}

func TestRkToolParserAcceptsEncodedQwenArguments(t *testing.T) {
	p := NewRkToolParser()
	_, calls := p.ProcessChunk("[TOOL_CALL]{\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"file_path\\\":\\\"main.go\\\"}\"}}[/TOOL_CALL]")
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	if got, want := string(calls[0].Function.Arguments), `{"file_path":"main.go"}`; got != want {
		t.Fatalf("arguments = %s, want %s", got, want)
	}
}

func TestRkToolParserHandlesMalformedCallAsToolError(t *testing.T) {
	p := NewRkToolParser()
	visible, calls := p.ProcessChunk("[TOOL_CALL]{\"function\":{\"name\":\"write_file\",\"arguments\":{\"content\":\"oops\"}}[/TOOL_CALL]")
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1 error call", len(calls))
	}
	if calls[0].Function.Name != "_error" {
		t.Fatalf("expected tool call name '_error', got %q", calls[0].Function.Name)
	}
	if visible != "" {
		t.Fatalf("expected 0 visible text (no raw tag leak), got %q", visible)
	}

	res := dispatchToolCall(calls[0].Function.Name, calls[0].Function.Arguments)
	if !strings.Contains(res, "Tool Error:") {
		t.Fatalf("expected dispatchToolCall to return Tool Error, got %q", res)
	}
}

func TestRkToolParserAcceptsExtraFields(t *testing.T) {
	p := NewRkToolParser()
	_, calls := p.ProcessChunk("[TOOL_CALL]{\"name\":\"get_devices\",\"arguments\":{},\"type\":\"function\",\"id\":\"call_1\"}[/TOOL_CALL]")
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	if calls[0].Function.Name != "get_devices" {
		t.Fatalf("unexpected call name: %s", calls[0].Function.Name)
	}
}

func TestRkToolParserReturnsMultipleCalls(t *testing.T) {
	p := NewRkToolParser()
	_, calls := p.ProcessChunk("[TOOL_CALL]{\"name\":\"get_devices\",\"arguments\":{}}[/TOOL_CALL][TOOL_CALL]{\"name\":\"get_devices\",\"arguments\":{}}[/TOOL_CALL]")
	if len(calls) != 2 {
		t.Fatalf("got %d calls, want 2", len(calls))
	}
}
