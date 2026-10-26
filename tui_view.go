package main

import (
	"fmt"
	"path/filepath"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
)

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

	cacheDisplay := "Cache: ⚡"
	if !loadConfig().RKLLMCacheEnabled {
		cacheDisplay = "Cache: off"
	}

	abortHint := "[Esc] Abort"
	if m.isStreaming {
		abortHint = "[Esc] ⏳ Generating..."
	}

	statusText := fmt.Sprintf(
		" Model: %s | Toolset: %s | %s | Ctx: %s [%s] %d%% | %s | /help",
		modelDisplay,
		GetActiveToolset(),
		cacheDisplay,
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
	{"/enablecache", "Enable prompt caching"},
	{"/disablecache", "Disable prompt caching"},
	{"/file", "Load a local file into context"},
	{"/status", "Check RKLLM backend status"},
	{"/quit", "Exit Scooter"},
}

func (m model) renderAutocompletePopup() string {
	val := strings.TrimSpace(m.textarea.Value())
	if !strings.HasPrefix(val, "/") {
		return ""
	}

	lowerVal := strings.ToLower(val)

	if strings.HasPrefix(lowerVal, "/models") {
		var matches []struct{ cmd, desc string }
		for _, c := range availableCommands {
			if strings.HasPrefix(strings.ToLower(c.cmd), lowerVal) {
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

	if strings.HasPrefix(lowerVal, "/model") {
		query := ""
		if strings.HasPrefix(lowerVal, "/model ") {
			query = strings.TrimSpace(val[len("/model "):])
		} else if len(val) > len("/model") {
			query = strings.TrimSpace(val[len("/model"):])
		}

		models := listRKLLMModels()
		var matches []string
		for _, mdl := range models {
			if query == "" || strings.Contains(strings.ToLower(mdl), strings.ToLower(query)) {
				matches = append(matches, mdl)
			}
		}

		if len(matches) == 0 {
			matches = models
		}

		var popupLines []string
		popupLines = append(popupLines, lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Bold(true).Render(" ── RKLLM Models ── "))
		for _, match := range matches {
			displayMatch := match
			if filepath.IsAbs(match) {
				displayMatch = filepath.Base(match) + " (" + match + ")"
			}
			line := fmt.Sprintf(" %s ", displayMatch)
			popupLines = append(popupLines, lipgloss.NewStyle().Foreground(lipgloss.Color("252")).Render(line))
		}

		popupBoxStyle := lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("63")).
			Background(lipgloss.Color("235")).
			Padding(0, 1)

		return popupBoxStyle.Render(strings.Join(popupLines, "\n"))
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

func (m *model) refreshViewport() {
	if m.viewport.Width > 0 {
		wrapped := lipgloss.NewStyle().Width(m.viewport.Width).Render(m.buildDisplayString())
		m.viewport.SetContent(wrapped)
	} else {
		m.viewport.SetContent(m.buildDisplayString())
	}
	m.viewport.GotoBottom()
}
