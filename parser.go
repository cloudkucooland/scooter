package main

import (
	"encoding/json"
	"strings"
)

const MaxBufferSize = 4096

type RkToolParser struct {
	buffer      string
	inToolCall  bool
	currentTool string
	params      map[string]string
}

func NewRkToolParser() *RkToolParser {
	return &RkToolParser{
		params: make(map[string]string),
	}
}

func (p *RkToolParser) Reset() {
	p.buffer = ""
	p.inToolCall = false
	p.currentTool = ""
	p.params = make(map[string]string)
}

func (p *RkToolParser) ProcessChunk(text string) (string, bool, ToolCall) {
	if len(p.buffer)+len(text) > MaxBufferSize {
		logDebug("Parser buffer limit reached (%d), resetting.", MaxBufferSize)
		p.Reset()
	}

	p.buffer += text
	visibleText := ""

	// 1. Detect start of tool call tag sequence
	if !p.inToolCall {
		if idx := strings.Index(p.buffer, "<tool_call>"); idx != -1 {
			p.inToolCall = true
			p.params = make(map[string]string)
			p.currentTool = ""

			// Render visible text that appeared prior to <tool_call>
			visibleText = p.buffer[:idx]
			p.buffer = p.buffer[idx:]
		} else {
			// Hold back trailing '<' to avoid leaking partial opening tags to the UI
			if lastLt := strings.LastIndex(p.buffer, "<"); lastLt != -1 && len(p.buffer)-lastLt < 11 {
				visibleText = p.buffer[:lastLt]
				p.buffer = p.buffer[lastLt:]
			} else {
				visibleText = p.buffer
				p.buffer = ""
			}
			return visibleText, false, ToolCall{}
		}
	}

	// 2. Process tool call parameters while inside <tool_call> block
	if p.inToolCall {
		// Extract function name
		if p.currentTool == "" {
			if start := strings.Index(p.buffer, "<function="); start != -1 {
				if end := strings.Index(p.buffer[start:], ">"); end != -1 {
					p.currentTool = p.buffer[start+10 : start+end]
					logDebug("Function name extracted: %s", p.currentTool)
				}
			}
		}

		// Iteratively scan and extract complete parameter value payloads
		for {
			pStart := strings.Index(p.buffer, "<parameter=")
			if pStart == -1 {
				break
			}
			pEnd := strings.Index(p.buffer[pStart:], ">")
			if pEnd == -1 {
				break
			}

			paramName := p.buffer[pStart+11 : pStart+pEnd]
			closeTag := "</parameter>"
			valEnd := strings.Index(p.buffer[pStart+pEnd:], closeTag)
			if valEnd == -1 {
				break // Incomplete chunk, await next stream token
			}

			valStart := pStart + pEnd + 1
			absoluteValEnd := pStart + pEnd + valEnd
			paramValue := strings.TrimSpace(p.buffer[valStart:absoluteValEnd])
			p.params[paramName] = paramValue
			logDebug("Parameter extracted: %s = %s", paramName, paramValue)

			// Safely remove parsed parameter from scratchpad
			p.buffer = p.buffer[:pStart] + p.buffer[absoluteValEnd+len(closeTag):]
		}

		// 3. Detect tool call termination block
		if closeIdx := strings.Index(p.buffer, "</tool_call>"); closeIdx != -1 {
			logDebug("Tool call closure detected")
			p.inToolCall = false

			jsonMap := make(map[string]interface{}, len(p.params))
			for k, v := range p.params {
				jsonMap[k] = v
			}
			rawArgs, _ := json.Marshal(jsonMap)

			tc := ToolCall{}
			tc.Function.Name = p.currentTool
			tc.Function.Arguments = json.RawMessage(rawArgs)
			logDebug("Tool call parsed: %s, Args: %s", tc.Function.Name, string(rawArgs))

			// Clean remaining buffer past the closing tag
			p.buffer = p.buffer[closeIdx+len("</tool_call>"):]
			p.currentTool = ""

			return visibleText, true, tc
		}
	}

	return visibleText, false, ToolCall{}
}

