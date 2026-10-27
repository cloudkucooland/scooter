package parser

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"codeberg.org/indievisible/scooter/pkg/types"
)

const (
	ToolCallStartTag = "[TOOL_CALL]"
	ToolCallEndTag   = "[/TOOL_CALL]"
	MaxToolCallSize  = 256 * 1024
)

var LogDebug = func(format string, v ...any) {}

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

func (p *RkToolParser) ProcessChunk(text string) (string, []types.ToolCall) {
	p.buffer.WriteString(text)

	var visible strings.Builder
	var calls []types.ToolCall

	for {
		content := p.buffer.String()
		if !p.inToolCall {
			start := strings.Index(content, ToolCallStartTag)
			if start == -1 {
				keep := partialTagSuffix(content, ToolCallStartTag)
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
			p.buffer.WriteString(content[start+len(ToolCallStartTag):])
			continue
		}

		end := strings.Index(content, ToolCallEndTag)
		if end == -1 {
			if len(content) > MaxToolCallSize {
				LogDebug("Tool call exceeded %d bytes; rendering it as text.", MaxToolCallSize)
				visible.WriteString(ToolCallStartTag)
				visible.WriteString(content)
				p.Reset()
			}
			return visible.String(), calls
		}

		rawCall := strings.TrimSpace(content[:end])
		remaining := content[end+len(ToolCallEndTag):]
		p.inToolCall = false
		p.buffer.Reset()
		p.buffer.WriteString(remaining)

		call, err := parseToolCall(rawCall)
		if err != nil {
			LogDebug("Invalid tool call: %v; payload: %s", err, rawCall)
			call = types.ToolCall{
				Function: struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				}{
					Name:      "_error",
					Arguments: json.RawMessage(fmt.Sprintf("%q", err.Error()+" in payload: "+rawCall)),
				},
			}
			calls = append(calls, call)
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

func parseToolCall(raw string) (types.ToolCall, error) {
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
	if err := decoder.Decode(&envelope); err != nil {
		return types.ToolCall{}, fmt.Errorf("invalid JSON: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return types.ToolCall{}, fmt.Errorf("tool call contains multiple JSON values")
		}
		return types.ToolCall{}, fmt.Errorf("invalid trailing JSON: %w", err)
	}

	name := envelope.Name
	arguments := envelope.Arguments
	if len(envelope.Function) > 0 {
		var function struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal(envelope.Function, &function); err != nil {
			return types.ToolCall{}, fmt.Errorf("invalid function object: %w", err)
		}
		name, arguments = function.Name, function.Arguments
	}
	if strings.TrimSpace(name) == "" {
		return types.ToolCall{}, fmt.Errorf("missing function name")
	}
	arguments = bytes.TrimSpace(arguments)
	if len(arguments) == 0 || bytes.Equal(arguments, []byte("null")) {
		arguments = json.RawMessage(`{}`)
	}
	if arguments[0] == '"' {
		var encoded string
		if err := json.Unmarshal(arguments, &encoded); err != nil {
			return types.ToolCall{}, fmt.Errorf("invalid encoded arguments: %w", err)
		}
		arguments = json.RawMessage(encoded)
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(arguments, &object); err != nil {
		return types.ToolCall{}, fmt.Errorf("arguments must be a JSON object: %w", err)
	}

	var call types.ToolCall
	call.Function.Name = name
	call.Function.Arguments = arguments
	return call, nil
}
