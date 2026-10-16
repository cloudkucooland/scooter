package main

import (
	"encoding/json"
)

type SummaryMessage struct {
	Summary  string
	Retained []Message
	Err      error
}

type ModelsMessage struct {
	Models []string
	Err    error
}

type ToolsetsMessage struct {
	Toolsets []string
	Err      error
}

type StatusMessage struct {
	Status string
	Err    error
}

type Message struct {
	Role    string `json:"role"` // "system", "user", "assistant", "tool"
	Content string `json:"content"`
}

type StreamMessage struct {
	Content   string
	ToolCalls []ToolCall
	Done      bool
	Err       error
}

type ToolCall struct {
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}
