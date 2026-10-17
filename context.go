package main

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync"
	"unicode/utf8"
)

const (
	conversationSummaryLabel = "Conversation summary:\n"
)

var (
	contextMu          sync.RWMutex
	activeContextLimit = getDefaultContextTokens()
	contextPathRegex   = regexp.MustCompile(`(?i)(?:ctx|[-_/])(\d{1,3})k\b`)
)

func getActiveContextLimit() int {
	contextMu.RLock()
	defer contextMu.RUnlock()
	return activeContextLimit
}

func setActiveContextLimit(tokens int) int {
	minTok := getMinContextTokens()
	maxTok := getMaxContextTokens()
	if tokens < minTok {
		tokens = minTok
	}
	if tokens > maxTok {
		tokens = maxTok
	}
	contextMu.Lock()
	activeContextLimit = tokens
	contextMu.Unlock()
	return tokens
}

func getResponseReserveTokens(limit int) int {
	if limit >= 16384 {
		return 1536
	} else if limit >= 8192 {
		return 1024
	}
	return 768
}

func parseContextSize(raw string) (int, error) {
	s := strings.TrimSpace(strings.ToLower(raw))
	s = strings.TrimPrefix(s, "ctx")
	if strings.HasSuffix(s, "k") {
		numStr := strings.TrimSuffix(s, "k")
		var k int
		if _, err := fmt.Sscanf(numStr, "%d", &k); err != nil || k <= 0 {
			return 0, fmt.Errorf("invalid context size %q", raw)
		}
		return k * 1024, nil
	}
	var tokens int
	if _, err := fmt.Sscanf(s, "%d", &tokens); err != nil || tokens <= 0 {
		return 0, fmt.Errorf("invalid context size %q", raw)
	}
	return tokens, nil
}

func detectContextLength(path string) int {
	minTok := getMinContextTokens()
	if env := os.Getenv("RKLLM_CONTEXT_LEN"); env != "" {
		if tokens, err := parseContextSize(env); err == nil && tokens >= minTok {
			return tokens
		}
	}
	if env := os.Getenv("RKLLM_CONTEXT"); env != "" {
		if tokens, err := parseContextSize(env); err == nil && tokens >= minTok {
			return tokens
		}
	}

	matches := contextPathRegex.FindStringSubmatch(path)
	if len(matches) > 1 {
		var k int
		if _, err := fmt.Sscanf(matches[1], "%d", &k); err == nil && k > 0 {
			return k * 1024
		}
	}

	return getDefaultContextTokens()
}

// estimateTokens deliberately overestimates normal English and source code.
// It is only used to leave safe room in a fixed-size model context.
func estimateTokens(text string) int {
	if text == "" {
		return 0
	}
	return (utf8.RuneCountInString(text) + 2) / 3
}

func estimateMessageTokens(msg Message) int {
	tokens := estimateTokens(msg.Role) + estimateTokens(msg.Content) + estimateTokens(msg.ToolName) + 4
	for _, call := range msg.ToolCalls {
		tokens += estimateTokens(toolCallSummaryText(call))
	}
	return tokens
}

func toolCallSummaryText(call ToolCall) string {
	arguments := call.Function.Arguments
	if len(arguments) == 0 {
		arguments = json.RawMessage(`{}`)
	}
	payload, err := json.Marshal(struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}{Name: call.Function.Name, Arguments: arguments})
	if err == nil {
		return string(payload)
	}
	return fmt.Sprintf("name=%q arguments=%q", call.Function.Name, string(arguments))
}

func historyNeedsCompression(history []Message) bool {
	limit := getActiveContextLimit()
	reserve := getResponseReserveTokens(limit)
	total := estimateTokens(buildToolInstructions(getSystemTools())) + 4
	for _, msg := range history {
		total += estimateMessageTokens(msg)
	}
	return total+reserve+getContextSafetyTokens() > limit
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

	limit := getActiveContextLimit()
	reserve := getResponseReserveTokens(limit)
	budget := limit - reserve - getContextSafetyTokens() - getSummaryReserveTokens()
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
		if msg.Role == "tool" && msg.ToolName != "" {
			sb.WriteString(" result from ")
			sb.WriteString(msg.ToolName)
		}
		sb.WriteString("] ")
		sb.WriteString(msg.Content)
		sb.WriteString("\n")
		for _, call := range msg.ToolCalls {
			sb.WriteString("[assistant tool call] ")
			sb.WriteString(toolCallSummaryText(call))
			sb.WriteString("\n")
		}
	}
	return sb.String()
}

/* func truncateToTokenBudget(text string, budget int) string {
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
} */

func truncateToTokenBudget(text string, budget int) string {
	if budget <= 0 {
		return "[content omitted for context budget]"
	}
	if estimateTokens(text) <= budget {
		return text
	}

	runes := []rune(text)
	maxRunes := budget * 3

	const notice = "\n[... truncated for context budget ...]\n"
	noticeRunes := utf8.RuneCountInString(notice)

	if maxRunes <= noticeRunes+4 {
		if len(runes) > maxRunes {
			return string(runes[:maxRunes])
		}
		return string(runes)
	}

	available := maxRunes - noticeRunes
	half := available / 2
	tail := available - half

	return string(runes[:half]) + notice + string(runes[len(runes)-tail:])
}
