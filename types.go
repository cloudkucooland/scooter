package main

import (
	"codeberg.org/indievisible/scooter/pkg/types"
)

type Message = types.Message
type StreamMessage = types.StreamMessage
type ToolCall = types.ToolCall

type SummaryMessage struct {
	Summary    string
	Retained   []Message
	Err        error
	Generation uint64
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

type ToolExecutionResult struct {
	Call   ToolCall
	Result string
}
