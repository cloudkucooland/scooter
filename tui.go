package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
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
	viewport       viewport.Model
	textarea       textarea.Model
	modelName      string
	contextLen     int
	chatHistory    []Message
	isStreaming    bool
	streamCancel   context.CancelFunc
	currentBuffer  *strings.Builder
	streamChan     chan StreamMessage
	pendingTools   []ToolCall
	confirmingTool *ToolCall
}

func (m *model) saveMessage(msg Message) {
	if err := saveMessageObject(msg); err != nil {
		errorMessage := Message{
			Role:    "system",
			Content: fmt.Sprintf("Cache error: could not save %s message: %v", msg.Role, err),
		}
		// The caller normally appends msg immediately before saving it. Keep
		// that message last so the UI remains in chronological order while the
		// persistence failure is still visible.
		m.chatHistory = append(m.chatHistory, errorMessage)
		if len(m.chatHistory) >= 2 && m.chatHistory[len(m.chatHistory)-2] == msg {
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

	m.refreshViewport()
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

func (m model) renderStatusBar() string {
	modelDisplay := filepath.Base(m.modelName)
	ctxLimit := getActiveContextLimit()
	currentTokens := calculateCurrentTokens(m.chatHistory, m.currentBuffer)

	ctxPercent := 0
	if ctxLimit > 0 {
		ctxPercent = (currentTokens * 100) / ctxLimit
		if ctxPercent > 100 {
			ctxPercent = 100
		}
	}

	barWidth := 10
	filled := (ctxPercent * barWidth) / 100
	if filled > barWidth {
		filled = barWidth
	}
	bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)

	tokenStr := fmt.Sprintf("%.1fk/%.1fk", float64(currentTokens)/1000.0, float64(ctxLimit)/1000.0)
	if ctxLimit < 1024 {
		tokenStr = fmt.Sprintf("%d/%dt", currentTokens, ctxLimit)
	}

	abortHint := "[Esc] Abort"
	if m.isStreaming {
		abortHint = "[Esc] ⏳ Generating..."
	}

	statusText := fmt.Sprintf(
		" Model: %s | Toolset: %s | Ctx: %s [%s] %d%% | %s | /help",
		modelDisplay,
		GetActiveToolset(),
		tokenStr,
		bar,
		ctxPercent,
		abortHint,
	)
	return statusBarStyle.Render(statusText)
}

var availableCommands = []struct {
	cmd  string
	desc string
}{
	{"/help", "Show help message"},
	{"/models", "List available RKLLM models"},
	{"/model", "Switch active RKLLM model"},
	{"/toolsets", "List available toolsets"},
	{"/toolset", "Switch active toolset (dev, home, all)"},
	{"/context", "View or change context size"},
	{"/compress", "Manually compress conversation history"},
	{"/clear", "Clear conversation history and cache"},
	{"/file", "Load a local file into context"},
	{"/status", "Check RKLLM backend status"},
	{"/quit", "Exit Scooter"},
}

func (m model) renderAutocompletePopup() string {
	val := strings.TrimSpace(m.textarea.Value())
	if !strings.HasPrefix(val, "/") {
		return ""
	}
	parts := strings.Fields(val)
	prefix := ""
	if len(parts) > 0 {
		prefix = strings.ToLower(parts[0])
	} else {
		prefix = "/"
	}

	var matches []struct{ cmd, desc string }
	for _, c := range availableCommands {
		if strings.HasPrefix(strings.ToLower(c.cmd), prefix) {
			matches = append(matches, c)
		}
	}

	if len(matches) == 0 {
		return ""
	}

	var popupLines []string
	popupLines = append(popupLines, lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Bold(true).Render(" ── Suggestions ── "))
	for _, match := range matches {
		line := fmt.Sprintf(" %-10s - %s ", match.cmd, match.desc)
		popupLines = append(popupLines, lipgloss.NewStyle().Foreground(lipgloss.Color("252")).Render(line))
	}

	popupBoxStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("63")).
		Background(lipgloss.Color("235")).
		Padding(0, 1)

	return popupBoxStyle.Render(strings.Join(popupLines, "\n"))
}

func (m model) View() string {
	statusBar := m.renderStatusBar()
	autocomplete := m.renderAutocompletePopup()

	middle := statusBar
	if autocomplete != "" {
		middle = autocomplete + "\n" + statusBar
	}

	return fmt.Sprintf(
		"%s\n%s\n%s\n%s",
		titleStyle.Render("Scooter Agent"),
		m.viewport.View(),
		middle,
		m.textarea.View(),
	)
}

func (m *model) buildDisplayString() string {
	var sb strings.Builder
	for _, msg := range m.chatHistory {
		switch msg.Role {
		case "user":
			sb.WriteString("User: " + msg.Content + "\n\n")
		case "assistant":
			sb.WriteString("AI: " + msg.Content + "\n\n")
		case "system":
			sb.WriteString("System: " + msg.Content + "\n\n")
		case "tool":
			lines := strings.Split(strings.TrimSpace(msg.Content), "\n")
			preview := ""
			if len(lines) > 0 {
				preview = lines[0]
				if len(preview) > 70 {
					preview = preview[:70] + "..."
				}
			}
			toolBoxStyle := lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(lipgloss.Color("42")).
				Foreground(lipgloss.Color("42")).
				Padding(0, 1)

			summary := fmt.Sprintf("⚙️ Tool Result [%d lines / %d chars]\n  %s", len(lines), len(msg.Content), preview)
			sb.WriteString(toolBoxStyle.Render(summary) + "\n\n")
		}
	}
	if m.currentBuffer.Len() > 0 {
		sb.WriteString("AI: " + m.currentBuffer.String() + "\n")
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
		if m.confirmingTool != nil {
			switch msg.String() {
			case "y", "Y":
				call := *m.confirmingTool
				m.confirmingTool = nil
				var args ShellArgs
				_ = json.Unmarshal(call.Function.Arguments, &args)

				logDebug("Dispatching confirmed shell tool: %s", args.Command)
				toolOutcomeText := dispatchToolCall(call.Function.Name, call.Function.Arguments)
				logDebug("Tool outcome: %s", toolOutcomeText)
				toolMsg := Message{Role: "tool", Content: toolOutcomeText}
				m.chatHistory = append(m.chatHistory, toolMsg)
				m.saveMessage(toolMsg)
				m.refreshViewport()

				return m, m.resumeOrProcessNextTools()

			case "n", "N", "esc", "q":
				m.confirmingTool = nil

				toolMsg := Message{Role: "tool", Content: "Tool execution cancelled by user."}
				m.chatHistory = append(m.chatHistory, toolMsg)
				m.saveMessage(toolMsg)
				m.refreshViewport()

				return m, m.resumeOrProcessNextTools()
			}
			return m, nil
		}

		switch msg.Type {
		case tea.KeyCtrlC:
			closeRKLLMSession()
			return m, tea.Quit
		case tea.KeyCtrlL:
			m.refreshViewport()
			return m, tea.ClearScreen
		case tea.KeyEsc:
			if m.isStreaming {
				abortRKLLM()
				if m.streamCancel != nil {
					m.streamCancel()
					m.streamCancel = nil
				}
				m.isStreaming = false
				partial := strings.TrimSpace(m.currentBuffer.String())
				m.currentBuffer.Reset()
				m.pendingTools = nil
				if partial != "" {
					aiResponse := Message{Role: "assistant", Content: partial + " [aborted]"}
					m.chatHistory = append(m.chatHistory, aiResponse)
					m.saveMessage(aiResponse)
				} else {
					m.chatHistory = append(m.chatHistory, Message{
						Role:    "system",
						Content: "[Prompt generation aborted by user]",
					})
				}
				m.refreshViewport()
				return m, nil
			}
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
		if !m.isStreaming {
			return m, nil
		}

		if msg.Err != nil {
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Error: " + msg.Err.Error()})
			m.refreshViewport()
			m.isStreaming = false
			if m.streamCancel != nil {
				m.streamCancel()
				m.streamCancel = nil
			}
			return m, nil
		}

		if msg.Done {
			if m.streamCancel != nil {
				m.streamCancel()
				m.streamCancel = nil
			}
			if m.currentBuffer.Len() > 0 {
				aiResponse := Message{Role: "assistant", Content: m.currentBuffer.String()}
				m.chatHistory = append(m.chatHistory, aiResponse)
				m.saveMessage(aiResponse)
				m.currentBuffer.Reset()
			}
			m.isStreaming = false

			// Execute tool calls sequentially with shell confirmation
			if len(m.pendingTools) > 0 {
				logDebug("Executing %d pending tool calls", len(m.pendingTools))
				return m, m.resumeOrProcessNextTools()
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
				m.saveMessage(historyMsg)
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

	case ToolsetsMessage:
		if msg.Err != nil {
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Toolsets Error: " + msg.Err.Error()})
		} else {
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Available Toolsets:\n- " + strings.Join(msg.Toolsets, "\n- ")})
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

	if strings.HasPrefix(input, "/help") {
		helpText := `Scooter Agent Commands:
- /help: Show this help message
- /models: List available RKLLM models
- /model <name>: Switch active RKLLM model
- /toolsets: List available toolsets
- /toolset <name>: Switch active toolset (development, home/iot, web, all)
- /context [size]: View or change context limit (e.g. /context 8k, /context 16384)
- /compress: Manually compress/summarize older conversation history
- /clear: Clear conversation history and session cache
- /file <path>: Load a local file into the conversation
- /status: Check RKLLM backend status
- /quit: Exit Scooter
- [Esc]: Abort generation or cancel pending tool confirmation`

		m.chatHistory = append(m.chatHistory, Message{
			Role:    "system",
			Content: helpText,
		})
		m.refreshViewport()
		return nil
	}

	if strings.HasPrefix(input, "/models") {
		return listModelsCmd()
	}

	if strings.HasPrefix(input, "/toolsets") {
		return listToolsetsCmd()
	}

	if strings.HasPrefix(input, "/status") {
		return statusCmd(m.modelName)
	}

	if strings.HasPrefix(input, "/model ") {
		newModel := strings.TrimSpace(strings.TrimPrefix(input, "/model "))
		resolved := getRKLLMModelPath(newModel)
		m.modelName = resolved
		detected := detectContextLength(resolved)
		m.contextLen = detected
		setActiveContextLimit(detected)
		m.chatHistory = append(m.chatHistory, Message{
			Role:    "system",
			Content: fmt.Sprintf("Switched model to %s (Context: %dk / %d tokens)", resolved, detected/1024, detected),
		})
		m.refreshViewport()
		return nil
	}

	if strings.HasPrefix(input, "/toolset ") {
		newToolset := strings.TrimSpace(strings.TrimPrefix(input, "/toolset "))
		if err := SetActiveToolset(newToolset); err != nil {
			m.chatHistory = append(m.chatHistory, Message{
				Role:    "system",
				Content: fmt.Sprintf("Error switching toolset: %v. Available: development, home, web, all", err),
			})
		} else {
			m.chatHistory = append(m.chatHistory, Message{
				Role:    "system",
				Content: fmt.Sprintf("Switched active toolset to %s", GetActiveToolset()),
			})
		}
		m.refreshViewport()
		return nil
	}

	if strings.HasPrefix(input, "/context") {
		arg := strings.TrimSpace(strings.TrimPrefix(input, "/context"))
		if arg == "" {
			used := 0
			for _, h := range m.chatHistory {
				used += estimateMessageTokens(h)
			}
			toolTokens := estimateTokens(buildToolInstructions(getSystemTools())) + 4
			totalUsed := used + toolTokens
			pct := 0
			if m.contextLen > 0 {
				pct = (totalUsed * 100) / m.contextLen
			}
			m.chatHistory = append(m.chatHistory, Message{
				Role: "system",
				Content: fmt.Sprintf("Context Configuration:\n- Limit: %d tokens (%dk)\n- Current conversation: %d tokens (%d%% of budget)\n- Tool instructions overhead: %d tokens\n- Use `/context <size>` (e.g. /context 4k, /context 8k, /context 16k, /context 16384) to adjust.",
					m.contextLen, m.contextLen/1024, used, pct, toolTokens),
			})
			m.refreshViewport()
			return nil
		}

		newSize, err := parseContextSize(arg)
		if err != nil {
			m.chatHistory = append(m.chatHistory, Message{
				Role:    "system",
				Content: fmt.Sprintf("Invalid context size %q. Examples: /context 4k, /context 8k, /context 16k, /context 16384", arg),
			})
			m.refreshViewport()
			return nil
		}

		m.contextLen = newSize
		setActiveContextLimit(newSize)
		m.chatHistory = append(m.chatHistory, Message{
			Role:    "system",
			Content: fmt.Sprintf("Context limit set to %d tokens (%dk). Next model run will allocate this buffer.", newSize, newSize/1024),
		})
		m.refreshViewport()
		return nil
	}

	if strings.HasPrefix(input, "/clear") {
		m.chatHistory = nil
		clearCache()

		if hydration := loadHydrationContext(); hydration != nil {
			m.chatHistory = append(m.chatHistory, *hydration)
			m.saveMessage(*hydration)
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
			m.saveMessage(fileMsg)
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
	m.saveMessage(userMsg)

	m.refreshViewport()

	m.isStreaming = true
	m.currentBuffer.Reset()
	m.streamChan = make(chan StreamMessage, 10)

	ctx, cancel := context.WithCancel(context.Background())
	m.streamCancel = cancel

	go streamRKLLM(ctx, m.modelName, m.chatHistory, m.streamChan)
	return waitForStream(m.streamChan)
}

func (m *model) resumeOrProcessNextTools() tea.Cmd {
	if len(m.pendingTools) > 0 {
		call := m.pendingTools[0]
		m.pendingTools = m.pendingTools[1:]

		if call.Function.Name == "shell" {
			var args ShellArgs
			_ = json.Unmarshal(call.Function.Arguments, &args)
			m.confirmingTool = &call
			msg := Message{
				Role:    "system",
				Content: fmt.Sprintf("Execute shell command?\n  $ %s\nPress [y] to confirm, [n] to cancel.", args.Command),
			}
			m.chatHistory = append(m.chatHistory, msg)
			m.saveMessage(msg)
			m.refreshViewport()
			return nil
		}

		logDebug("Dispatching tool: %s", call.Function.Name)
		toolOutcomeText := dispatchToolCall(call.Function.Name, call.Function.Arguments)
		logDebug("Tool outcome: %s", toolOutcomeText)
		toolMsg := Message{Role: "tool", Content: toolOutcomeText}
		m.chatHistory = append(m.chatHistory, toolMsg)
		m.saveMessage(toolMsg)
		m.refreshViewport()

		return m.resumeOrProcessNextTools()
	}

	// Loop tool result back to LLM context
	m.isStreaming = true
	m.streamChan = make(chan StreamMessage, 10)
	ctx, cancel := context.WithCancel(context.Background())
	m.streamCancel = cancel
	go streamRKLLM(ctx, m.modelName, m.chatHistory, m.streamChan)
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
