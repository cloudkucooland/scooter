package main

import (
	"strings"
	"unicode/utf8"
)

const (
	ModelContextTokens       = 4096
	ResponseReserveTokens    = 768
	ContextSafetyTokens      = 256
	SummaryReserveTokens     = 512
	SummaryInputTokenLimit   = 2048
	HydrationTokenLimit      = 1000
	conversationSummaryLabel = "Conversation summary:\n"
)

// estimateTokens deliberately overestimates normal English and source code.
// It is only used to leave safe room in a fixed-size model context.
func estimateTokens(text string) int {
	if text == "" {
		return 0
	}
	return (utf8.RuneCountInString(text) + 2) / 3
}

func estimateMessageTokens(msg Message) int {
	return estimateTokens(msg.Role) + estimateTokens(msg.Content) + 4
}

func historyNeedsCompression(history []Message) bool {
	total := estimateTokens(buildToolInstructions(getSystemTools())) + 4
	for _, msg := range history {
		total += estimateMessageTokens(msg)
	}
	return total+ResponseReserveTokens+ContextSafetyTokens > ModelContextTokens
}

// planCompression retains system context and as much recent conversation as
// fits, reserving room for a short durable summary of the older turns.
func planCompression(history []Message, force bool) (source, retained []Message) {
	var conversation []Message
	for _, msg := range history {
		if msg.Role == "system" && !strings.HasPrefix(msg.Content, conversationSummaryLabel) {
			retained = append(retained, msg)
			continue
		}
		conversation = append(conversation, msg)
	}

	if !force && !historyNeedsCompression(history) {
		return nil, history
	}
	if force && len(conversation) > 2 {
		source = append(source, conversation[:len(conversation)-2]...)
		retained = append(retained, conversation[len(conversation)-2:]...)
		return source, retained
	}

	budget := ModelContextTokens - ResponseReserveTokens - ContextSafetyTokens - SummaryReserveTokens
	budget -= estimateTokens(buildToolInstructions(getSystemTools())) + 4
	for _, msg := range retained {
		budget -= estimateMessageTokens(msg)
	}
	if budget < 0 {
		budget = 0
	}

	var recent []Message
	used := 0
	for i := len(conversation) - 1; i >= 0; i-- {
		msg := conversation[i]
		cost := estimateMessageTokens(msg)
		if used+cost <= budget {
			recent = append([]Message{msg}, recent...)
			used += cost
			continue
		}
		source = append(source, conversation[:i+1]...)
		break
	}

	// Keep the newest message even when it is a very large pasted file. The
	// model sees a clearly marked bounded excerpt rather than an overlong prompt.
	if len(recent) == 0 && len(conversation) > 0 {
		latest := conversation[len(conversation)-1]
		latest.Content = truncateToTokenBudget(latest.Content, budget)
		recent = append(recent, latest)
		source = conversation[:len(conversation)-1]
	}
	retained = append(retained, recent...)
	return source, retained
}

func renderMessages(messages []Message) string {
	var sb strings.Builder
	for _, msg := range messages {
		sb.WriteString("[")
		sb.WriteString(msg.Role)
		sb.WriteString("] ")
		sb.WriteString(msg.Content)
		sb.WriteString("\n")
	}
	return sb.String()
}

func truncateToTokenBudget(text string, budget int) string {
	if budget <= 0 {
		return "[content omitted for context budget]"
	}
	if estimateTokens(text) <= budget {
		return text
	}
	runes := []rune(text)
	maxRunes := budget * 3
	if maxRunes < 32 || len(runes) <= maxRunes {
		return string(runes[:min(len(runes), maxRunes)]) + "\n[truncated for context budget]"
	}
	half := (maxRunes - 32) / 2
	return string(runes[:half]) + "\n[... truncated for context budget ...]\n" + string(runes[len(runes)-half:])
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
