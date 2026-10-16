package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

const (
	DefaultModelName = "/home/scot/rkllama-models/qwen3-4b-16k/qwen3-4b.rkllm"
	PlaceholderText  = "Send a message... (Shift+Enter for newline)"
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
	viewport      viewport.Model
	textarea      textarea.Model
	modelName     string
	chatHistory   []Message
	isStreaming   bool
	currentBuffer *strings.Builder
	streamChan    chan StreamMessage
	pendingTools  []ToolCall
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

	m := model{
		textarea:      ta,
		viewport:      vp,
		modelName:     DefaultModelName,
		chatHistory:   chatHist,
		currentBuffer: &strings.Builder{},
	}

	m.refreshViewport()
	return m
}

func (m model) View() string {
	statusBar := statusBarStyle.Render(
		fmt.Sprintf(" Model: %s | /model <name> | /compress | /clear | /file <path> | /quit", m.modelName),
	)
	return fmt.Sprintf(
		"%s\n%s\n%s\n%s",
		titleStyle.Render("Scooter Agent"),
		m.viewport.View(),
		statusBar,
		m.textarea.View(),
	)
}

func (m *model) buildDisplayString() string {
	var sb strings.Builder
	for _, msg := range m.chatHistory {
		switch msg.Role {
		case "user":
			sb.WriteString("User: " + msg.Content + "\n")
		case "assistant":
			sb.WriteString("AI: " + msg.Content + "\n")
		case "system":
			sb.WriteString("System: " + msg.Content + "\n")
		case "tool":
			sb.WriteString("Tool: " + msg.Content + "\n")
		}
	}
	if m.currentBuffer.Len() > 0 {
		sb.WriteString("AI: " + m.currentBuffer.String())
	}
	return sb.String()
}

func (m *model) refreshViewport() {
	if m.viewport.Width > 0 {
		wrapped := lipgloss.NewStyle().
			Width(m.viewport.Width).
			Render(m.buildDisplayString())
		m.viewport.SetContent(wrapped)
	} else {
		m.viewport.SetContent(m.buildDisplayString())
	}
	m.viewport.GotoBottom()
}

func (m model) Init() tea.Cmd {
	return textarea.Blink
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var (
		tiCmd tea.Cmd
		vpCmd tea.Cmd
	)

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		headerHeight := 1
		statusBarHeight := 1
		textAreaHeight := 3

		m.viewport.Width = msg.Width
		m.viewport.Height = msg.Height - headerHeight - statusBarHeight - textAreaHeight
		m.viewport.Style = m.viewport.Style.Width(msg.Width)

		m.textarea.SetWidth(msg.Width)
		m.refreshViewport()
		return m, nil

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit
		case tea.KeyEnter:
			// Prevent submission while model is outputting
			if m.isStreaming {
				return m, nil
			}

			// Allow Shift+Enter for multiline inputs
			if msg.Alt {
				m.textarea, tiCmd = m.textarea.Update(msg)
				return m, tiCmd
			}

			input := strings.TrimSpace(m.textarea.Value())
			if input == "" {
				m.textarea.Reset()
				return m, nil
			}

			m.textarea.Reset()
			return m, m.handleCommand(input)
		}

	case StreamMessage:
		logDebug("Received StreamMessage: Content='%s', ToolCallsLen=%d, Done=%v", msg.Content, len(msg.ToolCalls), msg.Done)
		if msg.Err != nil {
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Error: " + msg.Err.Error()})
			m.refreshViewport()
			m.isStreaming = false
			return m, nil
		}

		if msg.Done {
			if m.currentBuffer.Len() > 0 {
				aiResponse := Message{Role: "assistant", Content: m.currentBuffer.String()}
				m.chatHistory = append(m.chatHistory, aiResponse)
				saveMessageObject(aiResponse)
				m.currentBuffer.Reset()
			}
			m.isStreaming = false

			// Execute tool calls sequentially
			if len(m.pendingTools) > 0 {
				logDebug("Executing %d pending tool calls", len(m.pendingTools))
				toolsToRun := m.pendingTools
				m.pendingTools = nil

				for _, call := range toolsToRun {
					logDebug("Dispatching tool: %s", call.Function.Name)
					toolOutcomeText := dispatchToolCall(call.Function.Name, call.Function.Arguments)
					logDebug("Tool outcome: %s", toolOutcomeText)
					toolMsg := Message{Role: "tool", Content: toolOutcomeText}
					m.chatHistory = append(m.chatHistory, toolMsg)
					saveMessageObject(toolMsg)
				}

				m.refreshViewport()

				// Loop tool result back to LLM context
				m.isStreaming = true
				m.streamChan = make(chan StreamMessage, 10)
				go streamOllama(context.Background(), m.modelName, m.chatHistory, m.streamChan)
				return m, waitForStream(m.streamChan)
			}

			m.refreshViewport()

			if historyNeedsCompression(m.chatHistory) {
				return m, compressHistoryCmd(m.modelName, m.chatHistory, false)
			}
			return m, nil
		}

		// Stream content accumulation
		m.currentBuffer.WriteString(msg.Content)
		if len(msg.ToolCalls) > 0 {
			m.pendingTools = append(m.pendingTools, msg.ToolCalls...)
		}

		m.refreshViewport()
		return m, waitForStream(m.streamChan)

	case SummaryMessage:
		if msg.Err != nil {
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Summary Error: " + msg.Err.Error()})
		} else {
			m.chatHistory = msg.Retained
			if msg.Summary != "" {
				summaryMsg := Message{Role: "system", Content: conversationSummaryLabel + msg.Summary}
				m.chatHistory = append([]Message{summaryMsg}, m.chatHistory...)
			}
			clearCache()
			for _, historyMsg := range m.chatHistory {
				saveMessageObject(historyMsg)
			}
		}
		m.refreshViewport()
		return m, nil

	case ModelsMessage:
		if msg.Err != nil {
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Models Error: " + msg.Err.Error()})
		} else {
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Available Models:\n" + strings.Join(msg.Models, "\n")})
		}
		m.refreshViewport()
		return m, nil

	case StatusMessage:
		if msg.Err != nil {
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Status Error: " + msg.Err.Error()})
		} else {
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: msg.Status})
		}
		m.refreshViewport()
		return m, nil
	}

	m.textarea, tiCmd = m.textarea.Update(msg)
	m.viewport, vpCmd = m.viewport.Update(msg)
	return m, tea.Batch(tiCmd, vpCmd)
}

func (m *model) handleCommand(input string) tea.Cmd {
	if strings.HasPrefix(input, "/quit") {
		closeRKLLMSession()
		return tea.Quit
	}

	if strings.HasPrefix(input, "/models") {
		return listModelsCmd()
	}

	if strings.HasPrefix(input, "/status") {
		return statusCmd()
	}

	if strings.HasPrefix(input, "/model ") {
		m.modelName = strings.TrimSpace(strings.TrimPrefix(input, "/model "))
		return nil
	}

	if strings.HasPrefix(input, "/clear") {
		m.chatHistory = nil
		clearCache()

		if hydration := loadHydrationContext(); hydration != nil {
			m.chatHistory = append(m.chatHistory, *hydration)
			saveMessageObject(*hydration)
		}

		m.refreshViewport()
		return nil
	}

	if strings.HasPrefix(input, "/file ") {
		filePath := strings.TrimSpace(strings.TrimPrefix(input, "/file "))
		content, err := os.ReadFile(filePath)
		if err != nil {
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Error reading file " + filePath + ": " + err.Error()})
		} else {
			fileMsg := Message{Role: "user", Content: fmt.Sprintf("File content (%s):\n%s", filePath, string(content))}
			m.chatHistory = append(m.chatHistory, fileMsg)
			saveMessageObject(fileMsg)
		}
		m.refreshViewport()
		return nil
	}

	if strings.HasPrefix(input, "/compress") {
		return compressHistoryCmd(m.modelName, m.chatHistory, true)
	}

	// Submit user input message
	userMsg := Message{Role: "user", Content: input}
	m.chatHistory = append(m.chatHistory, userMsg)
	saveMessageObject(userMsg)

	m.refreshViewport()

	m.isStreaming = true
	m.currentBuffer.Reset()
	m.streamChan = make(chan StreamMessage, 10)

	go streamOllama(context.Background(), m.modelName, m.chatHistory, m.streamChan)
	return waitForStream(m.streamChan)
}

func waitForStream(ch chan StreamMessage) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-ch
		if !ok {
			return StreamMessage{Done: true}
		}
		return msg
	}
}
