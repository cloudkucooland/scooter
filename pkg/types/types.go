package types

import "encoding/json"

type Message struct {
	Role      string     `json:"role"` // "system", "user", "assistant", "tool"
	Content   string     `json:"content"`
	ToolName  string     `json:"tool_name,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
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
