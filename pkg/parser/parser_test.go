package parser

import (
	"testing"
)

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
	_, calls := p.ProcessChunk("[TOOL_CALL]{\"function\":{\"name\":\"write_file\",\"arguments\":{\"content\":\"oops\"}}[/TOOL_CALL]")
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	if calls[0].Function.Name != "_error" {
		t.Fatalf("expected error tool name, got %q", calls[0].Function.Name)
	}
}

func TestRkToolParserAcceptsExtraFields(t *testing.T) {
	p := NewRkToolParser()
	_, calls := p.ProcessChunk("[TOOL_CALL]{\"name\":\"test\",\"arguments\":{},\"extra\":true}[/TOOL_CALL]")
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	if calls[0].Function.Name != "test" {
		t.Fatalf("expected test tool name, got %q", calls[0].Function.Name)
	}
}

func TestRkToolParserReturnsMultipleCalls(t *testing.T) {
	p := NewRkToolParser()
	_, calls := p.ProcessChunk("[TOOL_CALL]{\"name\":\"first\",\"arguments\":{}}[/TOOL_CALL] middle [TOOL_CALL]{\"name\":\"second\",\"arguments\":{}}[/TOOL_CALL]")
	if len(calls) != 2 {
		t.Fatalf("got %d calls, want 2", len(calls))
	}
	if calls[0].Function.Name != "first" || calls[1].Function.Name != "second" {
		t.Fatalf("unexpected calls order: %q, %q", calls[0].Function.Name, calls[1].Function.Name)
	}
}
