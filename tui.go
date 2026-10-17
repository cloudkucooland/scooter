package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
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
		// The caller normally appends msg immediately before saving it. Keep
		// that message last so the UI remains in chronological order while the
		// persistence failure is still visible.
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
			sb.WriteString("AI:\n")
			if msg.Content != "" {
				sb.WriteString(m.renderAssistantMessage(msg))
				sb.WriteString("\n")
			}
			for _, call := range msg.ToolCalls {
				sb.WriteString("Tool call: " + call.Function.Name + " " + string(call.Function.Arguments) + "\n")
			}
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

			toolLabel := "Tool Result"
			if msg.ToolName != "" {
				toolLabel += " [" + msg.ToolName + "]"
			}
			summary := fmt.Sprintf("⚙️ %s [%d lines / %d chars]\n  %s", toolLabel, len(lines), len(msg.Content), preview)
			sb.WriteString(toolBoxStyle.Render(summary) + "\n\n")
		}
	}
	if m.currentBuffer.Len() > 0 {
		sb.WriteString("AI: " + m.currentBuffer.String() + "\n")
	}
	return sb.String()
}

func (m *model) renderAssistantMessage(msg Message) string {
	if m.markdownWidth == m.viewport.Width {
		if rendered, ok := m.markdownCache[msg.Content]; ok {
			return rendered
		}
	}
	return msg.Content
}

func renderMarkdownCmd(messages []Message, width int) tea.Cmd {
	if width <= 0 {
		width = 80
	}
	toRender := append([]Message(nil), messages...)
	return func() tea.Msg {
		renderer, err := glamour.NewTermRenderer(
			glamour.WithStandardStyle("dark"),
			glamour.WithWordWrap(width),
		)
		if err != nil {
			return markdownRenderedMsg{width: width}
		}
		rendered := make(map[string]string)
		for _, msg := range toRender {
			if msg.Role != "assistant" {
				continue
			}
			formatted, err := renderer.Render(msg.Content)
			if err == nil {
				rendered[msg.Content] = formatted
			}
		}
		return markdownRenderedMsg{width: width, rendered: rendered}
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(textarea.Blink, renderMarkdownCmd(m.chatHistory, m.viewport.Width))
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var (
		tiCmd tea.Cmd
		vpCmd tea.Cmd
	)

	switch msg := msg.(type) {
	case markdownRenderedMsg:
		if msg.width != m.viewport.Width {
			return m, nil
		}
		if m.markdownCache == nil {
			m.markdownCache = make(map[string]string)
		}
		m.markdownWidth = msg.width
		for content, formatted := range msg.rendered {
			m.markdownCache[content] = formatted
		}
		m.refreshViewport()
		return m, nil

	case tea.WindowSizeMsg:
		headerHeight := 1
		statusBarHeight := 1
		textAreaHeight := 3

		m.viewport.Width = msg.Width
		m.viewport.Height = msg.Height - headerHeight - statusBarHeight - textAreaHeight
		m.viewport.Style = m.viewport.Style.Width(msg.Width)

		m.textarea.SetWidth(msg.Width)
		m.markdownWidth = -1
		m.markdownCache = make(map[string]string)
		return m, renderMarkdownCmd(m.chatHistory, msg.Width)

	case tea.KeyMsg:
		input := strings.TrimSpace(m.textarea.Value())
		isQuit := msg.Type == tea.KeyCtrlC ||
			(msg.Type == tea.KeyEnter && input != "" && strings.Fields(input)[0] == "/quit")
		if isQuit {
			m.pendingTools = nil
			m.confirmingTool = nil
			if m.isStreaming {
				cancel := m.streamCancel
				go func() {
					if cancel != nil {
						cancel()
					}
					abortRKLLM()
				}()
			}
			return m, tea.Quit
		}

		if m.confirmingTool != nil {
			switch msg.String() {
			case "y", "Y":
				call := *m.confirmingTool
				m.confirmingTool = nil
				var args ShellArgs
				_ = json.Unmarshal(call.Function.Arguments, &args)

				logDebug("Dispatching confirmed shell tool: %s", args.Command)
				m.isProcessingTool = true
				return m, runToolCallCmd(call)

			case "n", "N", "esc", "q":
				call := *m.confirmingTool
				m.confirmingTool = nil

				toolMsg := Message{Role: "tool", ToolName: call.Function.Name, Content: "Tool execution cancelled by user."}
				m.chatHistory = append(m.chatHistory, toolMsg)
				m.saveMessage(toolMsg)
				m.isProcessingTool = false
				m.refreshViewport()

				return m, m.resumeOrProcessNextTools()
			}
			return m, nil
		}

		switch msg.Type {
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
					m.refreshViewport()
					return m, renderMarkdownCmd([]Message{aiResponse}, m.viewport.Width)
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
			if msg.Alt {
				m.textarea, tiCmd = m.textarea.Update(msg)
				return m, tiCmd
			}

			input := strings.TrimSpace(m.textarea.Value())
			if input == "" {
				m.textarea.Reset()
				return m, nil
			}
			if m.isStreaming || m.isProcessingTool {
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
			m.isStreaming = false
			m.isProcessingTool = false
			if m.streamCancel != nil {
				m.streamCancel()
				m.streamCancel = nil
			}
			m.currentBuffer.Reset()
			m.pendingTools = nil
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Error: " + msg.Err.Error()})
			m.refreshViewport()
			return m, nil
		}

		if msg.Done {
			if m.streamCancel != nil {
				m.streamCancel()
				m.streamCancel = nil
			}
			var renderCmd tea.Cmd
			if m.currentBuffer.Len() > 0 || len(m.pendingTools) > 0 {
				aiResponse := Message{
					Role:      "assistant",
					Content:   m.currentBuffer.String(),
					ToolCalls: append([]ToolCall(nil), m.pendingTools...),
				}
				m.chatHistory = append(m.chatHistory, aiResponse)
				m.saveMessage(aiResponse)
				m.currentBuffer.Reset()
				renderCmd = renderMarkdownCmd([]Message{aiResponse}, m.viewport.Width)
			}
			m.isStreaming = false

			if len(m.pendingTools) > 0 {
				m.toolRounds++
				if m.toolRounds > maxToolRoundsPerTurn {
					m.pendingTools = nil
					m.chatHistory = append(m.chatHistory, Message{
						Role:    "system",
						Content: fmt.Sprintf("Stopped tool execution after %d rounds. Review the results and send a new prompt to continue.", maxToolRoundsPerTurn),
					})
					m.refreshViewport()
					return m, renderCmd
				}
				logDebug("Executing %d pending tool calls in round %d", len(m.pendingTools), m.toolRounds)
				return m, tea.Batch(renderCmd, m.resumeOrProcessNextTools())
			}

			m.refreshViewport()
			if historyNeedsCompression(m.chatHistory) {
				return m, tea.Batch(renderCmd, compressHistoryCmd(m.modelName, m.chatHistory, false))
			}
			return m, renderCmd
		}

		m.currentBuffer.WriteString(msg.Content)
		if len(msg.ToolCalls) > 0 {
			m.pendingTools = append(m.pendingTools, msg.ToolCalls...)
		}

		if !m.streamRenderPending {
			m.streamRenderPending = true
			return m, tea.Batch(waitForStream(m.streamChan), tea.Tick(streamRenderInterval, func(time.Time) tea.Msg {
				return streamRenderTick{}
			}))
		}
		return m, waitForStream(m.streamChan)

	case streamRenderTick:
		m.streamRenderPending = false
		if m.isStreaming {
			m.refreshViewport()
		}
		return m, nil

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

	case ToolExecutionResult:
		m.isProcessingTool = false
		toolMsg := Message{Role: "tool", ToolName: msg.Call.Function.Name, Content: msg.Result}
		m.chatHistory = append(m.chatHistory, toolMsg)
		m.saveMessage(toolMsg)
		m.refreshViewport()
		return m, m.resumeOrProcessNextTools()

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

func (m *model) refreshViewport() {
	if m.viewport.Width > 0 {
		wrapped := lipgloss.NewStyle().Width(m.viewport.Width).Render(m.buildDisplayString())
		m.viewport.SetContent(wrapped)
	} else {
		m.viewport.SetContent(m.buildDisplayString())
	}
	m.viewport.GotoBottom()
}

func (m *model) handleCommand(input string) tea.Cmd {
	if input == "/quit" {
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
	m.toolRounds = 0
	m.toolCallCount = 0
	m.seenToolCalls = make(map[string]bool)

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
	for len(m.pendingTools) > 0 {
		call := m.pendingTools[0]
		m.pendingTools = m.pendingTools[1:]
		if m.toolCallCount >= maxToolCallsPerTurn {
			m.pendingTools = nil
			m.chatHistory = append(m.chatHistory, Message{
				Role:    "system",
				Content: fmt.Sprintf("Stopped tool execution after %d tool calls in this turn.", maxToolCallsPerTurn),
			})
			m.refreshViewport()
			return nil
		}
		m.toolCallCount++
		key := toolCallDedupKey(call)
		if m.seenToolCalls == nil {
			m.seenToolCalls = make(map[string]bool)
		}
		if m.seenToolCalls[key] {
			toolMsg := Message{
				Role:     "tool",
				ToolName: call.Function.Name,
				Content:  "Duplicate call skipped within this turn.",
			}
			m.chatHistory = append(m.chatHistory, toolMsg)
			m.saveMessage(toolMsg)
			m.refreshViewport()
			continue
		}
		m.seenToolCalls[key] = true

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

		m.isProcessingTool = true
		return runToolCallCmd(call)
	}

	// Loop tool result back to LLM context
	m.isProcessingTool = false
	m.isStreaming = true
	m.streamChan = make(chan StreamMessage, 10)
	ctx, cancel := context.WithCancel(context.Background())
	m.streamCancel = cancel
	go streamRKLLM(ctx, m.modelName, m.chatHistory, m.streamChan)
	return waitForStream(m.streamChan)
}

func runToolCallCmd(call ToolCall) tea.Cmd {
	return func() tea.Msg {
		logDebug("Dispatching tool: %s", call.Function.Name)
		result := dispatchToolCall(call.Function.Name, call.Function.Arguments)
		logDebug("Tool outcome: %s", result)
		return ToolExecutionResult{Call: call, Result: result}
	}
}

func toolCallDedupKey(call ToolCall) string {
	arguments := call.Function.Arguments
	var decoded any
	if json.Unmarshal(arguments, &decoded) == nil {
		if normalized, err := json.Marshal(decoded); err == nil {
			arguments = normalized
		}
	}
	return call.Function.Name + "\x00" + string(arguments)
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
