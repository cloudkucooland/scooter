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
