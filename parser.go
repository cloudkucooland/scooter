package main

import (
	"encoding/json"
	"strings"
)

const MaxBufferSize = 4096

type RkToolParser struct {
	buffer       string
	inToolCall   bool
	currentTool  string
	params       map[string]string
}

func NewRkToolParser() *RkToolParser {
	return &RkToolParser{
		params: make(map[string]string),
	}
}

// Reset clears the parser state
func (p *RkToolParser) Reset() {
	p.buffer = ""
	p.inToolCall = false
	p.currentTool = ""
	p.params = make(map[string]string)
}

// ProcessChunk scans incoming raw text streams for custom RKLLM tags.
// Returns (textToDisplay, toolIsReady, structuredPayload)
func (p *RkToolParser) ProcessChunk(text string) (string, bool, ToolCall) {
	if len(p.buffer)+len(text) > MaxBufferSize {
		logDebug("Parser buffer exceeded limit (%d), resetting.", MaxBufferSize)
		p.Reset()
	}
	
	p.buffer += text
	logDebug("ProcessChunk received: %d bytes, buffer size: %d", len(text), len(p.buffer))
	
	visibleText := text

	// 1. Detect tool call initiation block
	if !p.inToolCall {
		if idx := strings.Index(p.buffer, "<tool_call>"); idx != -1 {
			p.inToolCall = true
			p.params = make(map[string]string)
			p.currentTool = ""
			
			if idxText := strings.Index(text, "<tool_call>"); idxText != -1 {
				visibleText = text[:idxText]
			} else {
				visibleText = ""
			}
		}
	} else {
		visibleText = ""
	}

	// 2. Isolate processing state while tokens unroll over the channel
	if p.inToolCall {
		// ... existing logic ...
		
		// 3. Extract Function Name if not already locked in
		if p.currentTool == "" {
			if start := strings.Index(p.buffer, "<function="); start != -1 {
				if end := strings.Index(p.buffer[start:], ">"); end != -1 {
					p.currentTool = p.buffer[start+10 : start+end]
					logDebug("Function name extracted: %s", p.currentTool)
				}
			}
		}

		// 4. Iteratively scan and extract complete parameter value payloads
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
			valEnd := strings.Index(p.buffer, closeTag)
			if valEnd == -1 {
				break // Content block is still arriving over the socket loop
			}

			valStart := pStart + pEnd + 1
			paramValue := strings.TrimSpace(p.buffer[valStart:valEnd])
			p.params[paramName] = paramValue
			logDebug("Parameter extracted: %s = %s", paramName, paramValue)

			// Discard the processed parameter segment from our scratchpad buffer
			p.buffer = p.buffer[valEnd+len(closeTag):]
		}

		// 5. Detect definitive closure sequence
		if strings.Contains(p.buffer, "</tool_call>") {
			logDebug("Tool call closure detected")
			p.inToolCall = false

			// Compile the parsed parameter string map directly into structural raw JSON bytes
			jsonMap := make(map[string]interface{})
			for k, v := range p.params {
				jsonMap[k] = v
			}
			rawArgs, _ := json.Marshal(jsonMap)

			tc := ToolCall{}
			tc.Function.Name = p.currentTool
			tc.Function.Arguments = json.RawMessage(rawArgs)
			logDebug("Tool call parsed: %s, Args: %s", tc.Function.Name, string(rawArgs))

			// Completely flush out the structure manager
			p.buffer = strings.Split(p.buffer, "</tool_call>")[1] // Keep remainder
			p.currentTool = ""

			return visibleText, true, tc
		}
	}

	return visibleText, false, ToolCall{}
}

