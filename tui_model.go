package main

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	"github.com/charmbracelet/lipgloss"
)

const (
	PlaceholderText = "Send a message... ([Esc] to stop generation)"
)

var (
	titleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("205")).
			Bold(true)
	statusBarStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "#343433", Dark: "#C7C6C5"}).
			Background(lipgloss.AdaptiveColor{Light: "#D9DCCF", Dark: "#353533"})
)

type model struct {
	viewport            viewport.Model
	textarea            textarea.Model
	modelName           string
	contextLen          int
	chatHistory         []Message
	isStreaming         bool
	streamCancel        context.CancelFunc
	currentBuffer       *strings.Builder
	streamChan          chan StreamMessage
	pendingTools        []ToolCall
	confirmingTool      *ToolCall
	markdownWidth       int
	markdownCache       map[string]string
	streamRenderPending bool
	isProcessingTool    bool
	toolRounds          int
	toolCallCount       int
	seenToolCalls       map[string]bool
	historyGeneration   uint64
}

const streamRenderInterval = 75 * time.Millisecond

const (
	maxToolRoundsPerTurn = 5
	maxToolCallsPerTurn  = 10
)

type streamRenderTick struct{}

type markdownRenderedMsg struct {
	width    int
	rendered map[string]string
}

func (m *model) saveMessage(msg Message) {
	if err := saveMessageObject(msg); err != nil {
		errorMessage := Message{
			Role:    "system",
			Content: fmt.Sprintf("Cache error: could not save %s message: %v", msg.Role, err),
		}
		m.chatHistory = append(m.chatHistory, errorMessage)
		if len(m.chatHistory) >= 2 && reflect.DeepEqual(m.chatHistory[len(m.chatHistory)-2], msg) {
			last := len(m.chatHistory) - 1
			m.chatHistory[last-1], m.chatHistory[last] = m.chatHistory[last], m.chatHistory[last-1]
		}
	}
}

func initialModel() model {
	ta := textarea.New()
	ta.Placeholder = PlaceholderText
	ta.Focus()
	ta.SetHeight(3)

	vp := viewport.New(80, 0)

	chatHist := loadHistoryObjects()

	if len(chatHist) == 0 {
		if hydration := loadHydrationContext(); hydration != nil {
			chatHist = append(chatHist, *hydration)
			saveMessageObject(*hydration)
		}
	}

	modelName := getDefaultModelName()
	initialCtx := detectContextLength(modelName)
	setActiveContextLimit(initialCtx)

	m := model{
		textarea:      ta,
		viewport:      vp,
		modelName:     modelName,
		contextLen:    initialCtx,
		chatHistory:   chatHist,
		currentBuffer: &strings.Builder{},
	}

	return m
}

func calculateCurrentTokens(history []Message, currentBuffer *strings.Builder) int {
	total := estimateTokens(buildToolInstructions(getSystemTools())) + 4
	for _, msg := range history {
		total += estimateMessageTokens(msg)
	}
	if currentBuffer != nil && currentBuffer.Len() > 0 {
		total += estimateTokens(currentBuffer.String())
	}
	return total
}
