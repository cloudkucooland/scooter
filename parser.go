package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

const (
	toolCallStartTag = "[TOOL_CALL]"
	toolCallEndTag   = "[/TOOL_CALL]"
	// Tool calls may contain a source file, but must still be bounded so a
	// malformed model response cannot grow memory without limit.
	MaxToolCallSize = 256 * 1024
)

type RkToolParser struct {
	buffer     strings.Builder
	inToolCall bool
}

func NewRkToolParser() *RkToolParser {
	return &RkToolParser{}
}

func (p *RkToolParser) Reset() {
	p.buffer.Reset()
	p.inToolCall = false
}

// ProcessChunk returns text safe to display and every complete, valid tool call
// found in the chunk. Invalid calls are returned as ordinary text so they can
// never be executed accidentally.
func (p *RkToolParser) ProcessChunk(text string) (string, []ToolCall) {
	p.buffer.WriteString(text)

	var visible strings.Builder
	var calls []ToolCall

	for {
		content := p.buffer.String()
		if !p.inToolCall {
			start := strings.Index(content, toolCallStartTag)
			if start == -1 {
				// Keep only a suffix that could become the opening marker in the
				// next streamed chunk. This prevents leaking split markers.
				keep := partialTagSuffix(content, toolCallStartTag)
				visible.WriteString(content[:len(content)-keep])
				p.buffer.Reset()
				if keep > 0 {
					p.buffer.WriteString(content[len(content)-keep:])
				}
				return visible.String(), calls
			}

			visible.WriteString(content[:start])
			p.inToolCall = true
			p.buffer.Reset()
			p.buffer.WriteString(content[start+len(toolCallStartTag):])
			continue
		}

		end := strings.Index(content, toolCallEndTag)
		if end == -1 {
			if len(content) > MaxToolCallSize {
				logDebug("Tool call exceeded %d bytes; rendering it as text.", MaxToolCallSize)
				visible.WriteString(toolCallStartTag)
				visible.WriteString(content)
				p.Reset()
			}
			return visible.String(), calls
		}

		rawCall := strings.TrimSpace(content[:end])
		remaining := content[end+len(toolCallEndTag):]
		p.inToolCall = false
		p.buffer.Reset()
		p.buffer.WriteString(remaining)

		call, err := parseToolCall(rawCall)
		if err != nil {
			// Do not repair or execute malformed JSON. Returning it to the model
			// transcript makes the failure observable on its next turn.
			logDebug("Invalid tool call: %v; payload: %s", err, rawCall)
			visible.WriteString(toolCallStartTag)
			visible.WriteString(rawCall)
			visible.WriteString(toolCallEndTag)
			continue
		}
		calls = append(calls, call)
	}
}

func partialTagSuffix(text, tag string) int {
	max := len(tag) - 1
	if len(text) < max {
		max = len(text)
	}
	for size := max; size > 0; size-- {
		if strings.HasSuffix(text, tag[:size]) {
			return size
		}
	}
	return 0
}

func parseToolCall(raw string) (ToolCall, error) {
	clean := strings.TrimSpace(raw)
	clean = strings.TrimPrefix(clean, "```json")
	clean = strings.TrimPrefix(clean, "```")
	clean = strings.TrimSuffix(strings.TrimSpace(clean), "```")

	var envelope struct {
		Function  json.RawMessage `json:"function"`
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	decoder := json.NewDecoder(strings.NewReader(clean))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&envelope); err != nil {
		return ToolCall{}, fmt.Errorf("invalid JSON: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return ToolCall{}, fmt.Errorf("tool call contains multiple JSON values")
		}
		return ToolCall{}, fmt.Errorf("invalid trailing JSON: %w", err)
	}

	name := envelope.Name
	arguments := envelope.Arguments
	if len(envelope.Function) > 0 {
		var function struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal(envelope.Function, &function); err != nil {
			return ToolCall{}, fmt.Errorf("invalid function object: %w", err)
		}
		name, arguments = function.Name, function.Arguments
	}
	if strings.TrimSpace(name) == "" {
		return ToolCall{}, fmt.Errorf("missing function name")
	}
	arguments = bytes.TrimSpace(arguments)
	if len(arguments) == 0 || bytes.Equal(arguments, []byte("null")) {
		arguments = json.RawMessage(`{}`)
	}
	if arguments[0] == '"' {
		var encoded string
		if err := json.Unmarshal(arguments, &encoded); err != nil {
			return ToolCall{}, fmt.Errorf("invalid encoded arguments: %w", err)
		}
		arguments = json.RawMessage(encoded)
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(arguments, &object); err != nil {
		return ToolCall{}, fmt.Errorf("arguments must be a JSON object: %w", err)
	}

	var call ToolCall
	call.Function.Name = name
	call.Function.Arguments = arguments
	return call, nil
}
