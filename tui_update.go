package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"codeberg.org/indievisible/scooter/pkg/knowledge"
	"codeberg.org/indievisible/scooter/pkg/rkllm"
	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"
)

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
					rkllm.AbortRKLLM()
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
				m.pendingTools = nil // Clear all remaining pending tools on cancellation

				toolMsg := Message{Role: "tool", ToolName: call.Function.Name, Content: "Tool execution cancelled by user."}
				m.chatHistory = append(m.chatHistory, toolMsg)
				m.saveMessage(toolMsg)
				m.isProcessingTool = false
				m.refreshViewport()

				return m, nil
			}
			return m, nil
		}

		switch msg.Type {
		case tea.KeyTab:
			val := m.textarea.Value()
			lowerVal := strings.ToLower(strings.TrimSpace(val))
			if strings.HasPrefix(lowerVal, "/model") && !strings.HasPrefix(lowerVal, "/models") {
				query := ""
				if strings.HasPrefix(lowerVal, "/model ") {
					query = strings.TrimSpace(val[len("/model "):])
				}
				models := rkllm.ListRKLLMModels()
				for _, mdl := range models {
					if query == "" || strings.Contains(strings.ToLower(mdl), strings.ToLower(query)) {
						m.textarea.SetValue("/model " + mdl)
						m.textarea.CursorEnd()
						return m, nil
					}
				}
			} else if strings.HasPrefix(lowerVal, "/") {
				var matches []string
				for _, c := range availableCommands {
					if strings.HasPrefix(strings.ToLower(c.cmd), lowerVal) {
						matches = append(matches, c.cmd)
					}
				}
				if len(matches) > 0 {
					m.textarea.SetValue(matches[0] + " ")
					m.textarea.CursorEnd()
					return m, nil
				}
			}
		case tea.KeyCtrlL:
			m.refreshViewport()
			return m, tea.ClearScreen
		case tea.KeyEsc:
			if m.isStreaming {
				rkllm.AbortRKLLM()
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
				return m, tea.Batch(renderCmd, compressHistoryCmd(m.modelName, m.chatHistory, false, m.historyGeneration))
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
		if msg.Generation != m.historyGeneration {
			return m, nil
		}
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

func (m *model) handleCommand(input string) tea.Cmd {
	if input == "/quit" {
		return tea.Quit
	}

	if strings.HasPrefix(input, "!") {
		cmdStr := strings.TrimSpace(strings.TrimPrefix(input, "!"))
		userMsg := Message{Role: "user", Content: input}
		m.chatHistory = append(m.chatHistory, userMsg)
		m.saveMessage(userMsg)

		var result string
		if cmdStr == "" {
			result = "System Error: command is required after !"
		} else {
			result = runShell(cmdStr)
		}

		toolMsg := Message{Role: "tool", ToolName: "shell", Content: result}
		m.chatHistory = append(m.chatHistory, toolMsg)
		m.saveMessage(toolMsg)

		m.refreshViewport()
		return nil
	}

	if strings.HasPrefix(input, "/help") {
		helpText := `Scooter Agent Commands:
- /help: Show this help message
- /models: List available RKLLM models
- /model <name>: Switch active RKLLM model
- /toolsets: List available toolsets
- /toolset <name>: Switch active toolset (development, research, web, all)
- /context [size]: View or change context limit (e.g. /context 8k, /context 16384)
- /compress: Manually compress/summarize older conversation history
- /clear: Clear conversation history and session cache
- /cleartasks: Clear and reset active task checklist
- /disablecache: Disable prompt caching and remove cache files
- /enablecache: Enable prompt caching for the next request
- /file <path>: Load a local file into the conversation
- /status: Check RKLLM backend status
- /quit: Exit Scooter
- !<command>: Execute a shell command directly (e.g. !ls, !/bin/ls)
- [Esc]: Abort generation or cancel pending tool confirmation`

		m.chatHistory = append(m.chatHistory, Message{
			Role:    "system",
			Content: helpText,
		})
		m.refreshViewport()
		return nil
	}

	if strings.HasPrefix(input, "/disablecache") {
		cfg := loadConfig()
		cfg.RKLLMCacheEnabled = false
		_ = saveConfig(cfg)
		removePromptCacheFiles()
		_ = os.Setenv("RKLLM_CACHE_ENABLED", "false")
		m.chatHistory = append(m.chatHistory, Message{
			Role:    "system",
			Content: "Prompt caching disabled (`rkllm_cache_enabled` set to false) and existing cache files removed.",
		})
		m.refreshViewport()
		return nil
	}

	if strings.HasPrefix(input, "/enablecache") {
		cfg := loadConfig()
		cfg.RKLLMCacheEnabled = true
		_ = saveConfig(cfg)
		_ = os.Setenv("RKLLM_CACHE_ENABLED", "true")

		m.chatHistory = append(m.chatHistory, Message{
			Role:    "system",
			Content: "Prompt caching enabled (`rkllm_cache_enabled` set to true); the cache will be built for the current system prompt on the next request.",
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
		resolved := rkllm.GetRKLLMModelPath(newModel)
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
			removePromptCacheFiles()
			m.chatHistory = append(m.chatHistory, Message{
				Role:    "system",
				Content: fmt.Sprintf("Switched active toolset to %s (invalidated old prompt cache)", GetActiveToolset()),
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
		m.historyGeneration++
		m.chatHistory = nil
		m.currentBuffer.Reset()
		m.pendingTools = nil
		m.toolRounds = 0
		m.toolCallCount = 0
		m.seenToolCalls = nil
		clearCache()

		if hydration := loadHydrationContext(); hydration != nil {
			m.chatHistory = append(m.chatHistory, *hydration)
			m.saveMessage(*hydration)
		}

		m.refreshViewport()
		return nil
	}

	if strings.HasPrefix(input, "/cleartasks") {
		if err := ClearTasks(); err != nil {
			m.chatHistory = append(m.chatHistory, Message{
				Role:    "system",
				Content: fmt.Sprintf("Error clearing tasks: %v", err),
			})
		} else {
			m.chatHistory = append(m.chatHistory, Message{
				Role:    "system",
				Content: "Task checklist cleared successfully.",
			})
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

	if strings.HasPrefix(input, "/ingest ") {
		filePath := strings.TrimSpace(strings.TrimPrefix(input, "/ingest "))
		kdb, err := knowledge.Open()
		if err != nil {
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Error opening knowledge db: " + err.Error()})
		} else {
			defer kdb.Close()
			docID, err := kdb.IngestFile(filePath, "main")
			if err != nil {
				m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Error ingesting file " + filePath + ": " + err.Error()})
			} else {
				m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: fmt.Sprintf("Successfully ingested %s (Doc ID: %s) into knowledge base.", filePath, docID)})
			}
		}
		m.refreshViewport()
		return nil
	}

	if strings.HasPrefix(input, "/knowledge ") {
		query := strings.TrimSpace(strings.TrimPrefix(input, "/knowledge "))
		kdb, err := knowledge.Open()
		if err != nil {
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Error opening knowledge db: " + err.Error()})
		} else {
			defer kdb.Close()
			results, err := kdb.Search(query, "", 5)
			if err != nil {
				m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Knowledge search error: " + err.Error()})
			} else if len(results) == 0 {
				m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "No matching knowledge found for: " + query})
			} else {
				var sb strings.Builder
				sb.WriteString(fmt.Sprintf("Knowledge search results for %q:\n\n", query))
				for i, r := range results {
					sb.WriteString(fmt.Sprintf("[%d] **%s** (%s)\n%s\n\n", i+1, r.Title, r.Collection, r.Content))
				}
				m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: sb.String()})
			}
		}
		m.refreshViewport()
		return nil
	}

	if strings.HasPrefix(input, "/compress") {
		return compressHistoryCmd(m.modelName, m.chatHistory, true, m.historyGeneration)
	}

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

	go rkllm.StreamRKLLM(ctx, m.modelName, m.chatHistory, m.streamChan)
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

	m.isProcessingTool = false
	m.isStreaming = true
	m.streamChan = make(chan StreamMessage, 10)
	ctx, cancel := context.WithCancel(context.Background())
	m.streamCancel = cancel
	go rkllm.StreamRKLLM(ctx, m.modelName, m.chatHistory, m.streamChan)
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

func removePromptCacheFiles() {
	cacheDir, err := getCacheFile()
	if err != nil {
		return
	}
	dir := filepath.Dir(cacheDir)
	matches, err := filepath.Glob(filepath.Join(dir, "*.rkllm_cache"))
	if err != nil {
		return
	}
	for _, m := range matches {
		_ = os.Remove(m)
	}
}
