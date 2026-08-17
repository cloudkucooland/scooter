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
	DefaultModelName = "qwen3.5-4b"
	PlaceholderText  = "Send a message..."
	MaxHistorySize   = 30
)

// Styling
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
	pendingTools   []ToolCall 
}

func initialModel() model {
	ta := textarea.New()
	ta.Placeholder = PlaceholderText
	ta.Focus()
	ta.SetHeight(3)

	vp := viewport.New(80, 0)
	vp.Style = vp.Style.Width(80)

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
	statusBar := statusBarStyle.Render(fmt.Sprintf(" Model: %s | /model <name> | /quit | /compress | /clear | /file <path>", m.modelName))
	return fmt.Sprintf(
		"%s\n%s\n%s\n%s",
		titleStyle.Render("AI Agent TUI"),
		m.viewport.View(),
		statusBar,
		m.textarea.View(),
	)
}

func (m *model) renderDisplayLines() []string {
	var lines []string
	for _, msg := range m.chatHistory {
		switch msg.Role {
		case "user":
			lines = append(lines, "User: "+msg.Content)
		case "assistant":
			lines = append(lines, "AI: "+msg.Content)
		case "system":
			lines = append(lines, "System: "+msg.Content)
		case "tool":
			lines = append(lines, "Tool: "+msg.Content)
		}
	}
	return lines
}

func (m *model) refreshViewport() {
	lines := m.renderDisplayLines()
	content := strings.Join(lines, "\n")
	m.viewport.SetContent(content)
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

	m.textarea, tiCmd = m.textarea.Update(msg)
	m.viewport, vpCmd = m.viewport.Update(msg)

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		// Viewport takes remaining space: total height - title (1) - status bar (1) - textarea (3)
		headerHeight := 1
		statusBarHeight := 1
		textAreaHeight := 3
		m.viewport.Width = msg.Width
		m.viewport.Height = msg.Height - headerHeight - statusBarHeight - textAreaHeight
		m.textarea.SetWidth(msg.Width)
		m.viewport.Style = m.viewport.Style.Width(msg.Width)
		m.textarea.SetWidth(msg.Width)

		case StreamMessage:
		if msg.Err != nil {
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Error: " + msg.Err.Error()})
			m.refreshViewport()
			m.isStreaming = false
			return m, nil
		}

		if msg.Done {
			// Save the final text assistant response if any text accompanied the payload
			aiResponse := Message{Role: "assistant", Content: m.currentBuffer.String()}
			m.chatHistory = append(m.chatHistory, aiResponse)
			saveMessageObject(aiResponse)
			m.currentBuffer.Reset()
			m.isStreaming = false

			logDebug("Checking pending tools: %d", len(m.pendingTools))

			// CORE AGENT LOOP ROUTER: Check if rkllama requested any tool executions
			if len(m.pendingTools) > 0 {
				toolsToRun := m.pendingTools
				m.pendingTools = nil // Flush the buffer state immediately

				// Process the stacked tool execution calls sequentially in Go
				for _, call := range toolsToRun {
					logDebug("Executing tool: %s", call.Function.Name)
					toolOutcomeText := dispatchToolCall(call.Function.Name, call.Function.Arguments)
					
					// Append the outcome log message using the proper 'tool' context role tag
					toolMsg := Message{Role: "tool", Content: toolOutcomeText}
					m.chatHistory = append(m.chatHistory, toolMsg)
					saveMessageObject(toolMsg)
				}
				
				m.refreshViewport()

				// Auto-trigger a clean recursive loop back to rkllama with the new tool output
				m.isStreaming = true
				m.streamChan = make(chan StreamMessage, 10)
				go streamOllama(context.Background(), m.modelName, m.chatHistory, m.streamChan)
				return m, waitForStream(m.streamChan)
			}

			m.refreshViewport()

			if len(m.chatHistory) >= 30 {
				return m, summarizeCmd(m.modelName, m.chatHistory)
			}
			return m, nil
		}

		// Keep aggregating regular markdown chat body output
		m.currentBuffer.WriteString(msg.Content)
		
		// Add new tool calls to pending queue
		for _, tc := range msg.ToolCalls {
			logDebug("Tool call received in stream: %s", tc.Function.Name)
			m.pendingTools = append(m.pendingTools, tc)
		}

		// Render streaming terminal preview live
		displayLines := m.renderDisplayLines()
		if m.currentBuffer.Len() > 0 {
			displayLines = append(displayLines, "AI: "+m.currentBuffer.String())
		}
		m.viewport.SetContent(strings.Join(displayLines, "\n"))
		m.viewport.GotoBottom()
		return m, waitForStream(m.streamChan)

	case SummaryMessage:
		if msg.Err != nil {
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Summary Error: " + msg.Err.Error()})
		} else {
			summaryMsg := Message{Role: "assistant", Content: "Summary: " + msg.Summary}
			m.chatHistory = []Message{summaryMsg}
			clearCache()
			saveMessageObject(summaryMsg)
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

	case ToolCall:
			logDebug("Tool call received: %s", msg.Function.Name)
			m.pendingTools = append(m.pendingTools, msg)
			switch msg.Function.Name {
			case "write_file":
				m.chatHistory = append(m.chatHistory, Message{Role: "tool_call", Content: "write_file"})
				// do something
			case "set_smart_device":
				m.chatHistory = append(m.chatHistory, Message{Role: "tool_call", Content: "set_smart_device"})
				// do something
			default:
				m.chatHistory = append(m.chatHistory, Message{Role: "tool_call", Content: msg.Function.Name})
			}

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit
		case tea.KeyEnter:
			if m.isStreaming {
				return m, nil
			}

			input := strings.TrimSpace(m.textarea.Value())
			if input == "" {
				return m, nil
			}
			return m, m.handleCommand(input)
		}
	}

	return m, tea.Batch(tiCmd, vpCmd)
}

func (m *model) handleCommand(input string) tea.Cmd {
	if strings.HasPrefix(input, "/quit") {
		return tea.Quit
	}

	if strings.HasPrefix(input, "/models") {
		m.textarea.Reset()
		return listModelsCmd()
	}

	if strings.HasPrefix(input, "/status") {
		m.textarea.Reset()
		return statusCmd()
	}

	if strings.HasPrefix(input, "/model ") {
		m.modelName = strings.TrimSpace(strings.TrimPrefix(input, "/model "))
		m.textarea.Reset()
		return nil
	}

	if strings.HasPrefix(input, "/clear") {
		m.chatHistory = nil
		clearCache()

		// Re-hydrate if system context file is present
		if hydration := loadHydrationContext(); hydration != nil {
			m.chatHistory = append(m.chatHistory, *hydration)
			saveMessageObject(*hydration)
		}

		m.refreshViewport()
		m.textarea.Reset()
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
		m.textarea.Reset()
		return nil
	}

	if strings.HasPrefix(input, "/compress") {
		m.textarea.Reset()
		return summarizeCmd(m.modelName, m.chatHistory)
	}

	// Process normal prompt
	userMsg := Message{Role: "user", Content: input}
	m.chatHistory = append(m.chatHistory, userMsg)
	saveMessageObject(userMsg)

	m.refreshViewport()
	m.textarea.Reset()

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
