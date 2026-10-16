package main

import (
	"strings"
	"testing"
)

func TestPlanCompressionRetainsNewestConversationSuffix(t *testing.T) {
	history := []Message{{Role: "system", Content: "project instructions"}}
	for i := 0; i < 12; i++ {
		history = append(history, Message{Role: "user", Content: strings.Repeat("older context ", 250)})
	}
	history = append(history, Message{Role: "user", Content: "latest request"})

	source, retained := planCompression(history, false)
	if len(source) == 0 {
		t.Fatal("expected older history to be selected for summarization")
	}
	if got := retained[len(retained)-1].Content; got != "latest request" {
		t.Fatalf("latest message = %q, want it retained", got)
	}
	if retained[0].Role != "system" {
		t.Fatal("system context was not retained")
	}
}

func TestTruncateToTokenBudgetKeepsBoundedExcerpt(t *testing.T) {
	text := "begin " + strings.Repeat("middle ", 300) + "end"
	truncated := truncateToTokenBudget(text, 40)
	if estimateTokens(truncated) > 60 {
		t.Fatalf("truncated text is still too large: %d tokens", estimateTokens(truncated))
	}
	if !strings.Contains(truncated, "truncated for context budget") {
		t.Fatal("truncation marker missing")
	}
}

func TestParseContextSize(t *testing.T) {
	tests := []struct {
		input   string
		want    int
		wantErr bool
	}{
		{"4k", 4096, false},
		{"4K", 4096, false},
		{"8k", 8192, false},
		{"16k", 16384, false},
		{"32k", 32768, false},
		{"ctx16k", 16384, false},
		{"16384", 16384, false},
		{"4096", 4096, false},
		{"invalid", 0, true},
		{"0k", 0, true},
		{"-4k", 0, true},
	}

	for _, tt := range tests {
		got, err := parseContextSize(tt.input)
		if tt.wantErr {
			if err == nil {
				t.Errorf("parseContextSize(%q) expected error, got %d", tt.input, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseContextSize(%q) unexpected error: %v", tt.input, err)
			continue
		}
		if got != tt.want {
			t.Errorf("parseContextSize(%q) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

func TestDetectContextLength(t *testing.T) {
	tests := []struct {
		path string
		want int
	}{
		{"/home/scot/rkllama-models/qwen3-4b-16k/qwen3-4b.rkllm", 16384},
		{"gemma-4-E2B-it-RKLLM-RK3588-W8A8-ctx16k", 16384},
		{"models/deepseek-8k.rkllm", 8192},
		{"models/qwen3.5-4b/qwen.rkllm", 4096}, // 4b is parameters, not context k
		{"custom_model_32k.rkllm", 32768},
		{"unspecified-model.rkllm", 4096},
	}

	for _, tt := range tests {
		got := detectContextLength(tt.path)
		if got != tt.want {
			t.Errorf("detectContextLength(%q) = %d, want %d", tt.path, got, tt.want)
		}
	}
}

func TestDynamicCompressionBudget(t *testing.T) {
	origLimit := getActiveContextLimit()
	defer setActiveContextLimit(origLimit)

	setActiveContextLimit(4096)
	history := []Message{{Role: "system", Content: "instructions"}}
	// Add ~3000 tokens
	for i := 0; i < 6; i++ {
		history = append(history, Message{Role: "user", Content: strings.Repeat("test message text ", 100)})
	}

	// At 4096 tokens, ~3000 tokens of history + reserves will need compression
	needs4k := historyNeedsCompression(history)
	if !needs4k {
		t.Errorf("expected 4k context to need compression for 3000 tokens of history")
	}

	// At 16384 tokens, 3000 tokens of history fits comfortably with plenty of room
	setActiveContextLimit(16384)
	needs16k := historyNeedsCompression(history)
	if needs16k {
		t.Errorf("expected 16k context to not need compression for 3000 tokens of history")
	}
}
