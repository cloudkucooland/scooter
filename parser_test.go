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

func TestRkToolParserDoesNotExecuteMalformedCall(t *testing.T) {
	p := NewRkToolParser()
	visible, calls := p.ProcessChunk("[TOOL_CALL]{\"function\":{\"name\":\"write_file\",\"arguments\":{\"content\":\"oops\"}}[/TOOL_CALL]")
	if len(calls) != 0 {
		t.Fatalf("got %d calls, want 0", len(calls))
	}
	if visible == "" {
		t.Fatal("malformed call was discarded instead of rendered safely")
	}
}

func TestRkToolParserReturnsMultipleCalls(t *testing.T) {
	p := NewRkToolParser()
	_, calls := p.ProcessChunk("[TOOL_CALL]{\"name\":\"get_devices\",\"arguments\":{}}[/TOOL_CALL][TOOL_CALL]{\"name\":\"get_devices\",\"arguments\":{}}[/TOOL_CALL]")
	if len(calls) != 2 {
		t.Fatalf("got %d calls, want 2", len(calls))
	}
}
