package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
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

// Message matches the Ollama JSON wire protocol and is used for session persistence
type Message struct {
	Role    string `json:"role"` // "system", "user", "assistant"
	Content string `json:"content"`
}

type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
}

type ChatResponseChunk struct {
	Message Message `json:"message"`
	Done    bool    `json:"done"`
}

type StreamMessage struct {
	Content string
	Done    bool
	Err     error
}

type SummaryMessage struct {
	Summary string
	Err     error
}

func getOllamaHost() string {
	if host := os.Getenv("OLLAMA_HOST"); host != "" {
		return strings.TrimRight(host, "/")
	}
	return "http://localhost:11434"
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

// Session Persistence using JSON Lines (.jsonl)
func getCacheFile() (string, error) {
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	agentDir := fmt.Sprintf("%s/agent", cacheDir)
	if err := os.MkdirAll(agentDir, 0755); err != nil {
		return "", err
	}
	return fmt.Sprintf("%s/session.jsonl", agentDir), nil
}

func clearCache() {
	cacheFile, err := getCacheFile()
	if err != nil {
		return
	}
	os.Remove(cacheFile)
}

// Save a single discrete Message object as JSON
func saveMessageObject(msg Message) {
	cacheFile, err := getCacheFile()
	if err != nil {
		return
	}
	f, err := os.OpenFile(cacheFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	data, err := json.Marshal(msg)
	if err != nil {
		return
	}
	f.Write(data)
	f.WriteString("\n")
}

// Load structured JSON history safely
func loadHistoryObjects() []Message {
	cacheFile, err := getCacheFile()
	if err != nil {
		return nil
	}
	f, err := os.Open(cacheFile)
	if err != nil {
		return nil
	}
	defer f.Close()

	var history []Message
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var msg Message
		if err := json.Unmarshal(line, &msg); err == nil {
			history = append(history, msg)
		}
	}
	return history
}

// Loads system hydration context if AGENT.md or GEMINI.md exists locally
func loadHydrationContext() *Message {
	files := []string{"AGENT.md", "GEMINI.md"}
	for _, file := range files {
		content, err := os.ReadFile(file)
		if err == nil && len(strings.TrimSpace(string(content))) > 0 {
			return &Message{
				Role:    "system",
				Content: fmt.Sprintf("Context loaded from %s:\n%s", file, string(content)),
			}
		}
	}
	return nil
}

func summarizeCmd(modelName string, messages []Message) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Minute)
		defer cancel()

		var sb strings.Builder
		for _, m := range messages {
			sb.WriteString(fmt.Sprintf("[%s]: %s\n", m.Role, m.Content))
		}

		fullURL := getOllamaHost() + "/api/generate"
		prompt := fmt.Sprintf("Summarize the following conversation concisely:\n%s", sb.String())
		reqBody := map[string]interface{}{
			"model":  modelName,
			"prompt": prompt,
			"stream": false,
		}

		jsonData, _ := json.Marshal(reqBody)
		req, err := http.NewRequestWithContext(ctx, "POST", fullURL, bytes.NewBuffer(jsonData))
		if err != nil {
			return SummaryMessage{Err: err}
		}
		req.Header.Set("Content-Type", "application/json")

		client := &http.Client{}
		resp, err := client.Do(req)
		if err != nil {
			return SummaryMessage{Err: err}
		}
		defer resp.Body.Close()

		var response struct {
			Response string `json:"response"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
			return SummaryMessage{Err: err}
		}
		return SummaryMessage{Summary: response.Response}
	}
}

func streamOllama(ctx context.Context, modelName string, history []Message, channel chan<- StreamMessage) {
	defer close(channel)

	fullURL := getOllamaHost() + "/api/chat"
	reqBody := ChatRequest{
		Model:    modelName,
		Messages: history,
		Stream:   true,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		channel <- StreamMessage{Err: err}
		return
	}

	req, err := http.NewRequestWithContext(ctx, "POST", fullURL, bytes.NewBuffer(jsonData))
	if err != nil {
		channel <- StreamMessage{Err: err}
		return
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		channel <- StreamMessage{Err: err}
		return
	}
	defer resp.Body.Close()

	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			break
		}
		var chunk ChatResponseChunk
		if err := json.Unmarshal(line, &chunk); err == nil {
			channel <- StreamMessage{Content: chunk.Message.Content, Done: chunk.Done}
			if chunk.Done {
				break
			}
		}
	}
}

type model struct {
	viewport      viewport.Model
	textarea      textarea.Model
	modelName     string
	chatHistory   []Message
	isStreaming   bool
	currentBuffer strings.Builder
	streamChan    chan StreamMessage
}

func initialModel() model {
	ta := textarea.New()
	ta.Placeholder = "Send a message..."
	ta.Focus()
	ta.SetHeight(3)

	vp := viewport.New(0, 0)

	// Restore deterministic JSON session
	chatHist := loadHistoryObjects()

	// If no prior session, attempt default hydration file load
	if len(chatHist) == 0 {
		if hydration := loadHydrationContext(); hydration != nil {
			chatHist = append(chatHist, *hydration)
			saveMessageObject(*hydration)
		}
	}

	m := model{
		textarea:    ta,
		viewport:    vp,
		modelName:   "qwen3.5-4b",
		chatHistory: chatHist,
	}

	m.refreshViewport()
	return m
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
		}
	}
	return lines
}

func (m *model) refreshViewport() {
	lines := m.renderDisplayLines()
	m.viewport.SetContent(strings.Join(lines, "\n"))
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
		m.viewport.Width = msg.Width
		m.viewport.Height = msg.Height - 6
		m.textarea.SetWidth(msg.Width)

	case StreamMessage:
		if msg.Err != nil {
			m.chatHistory = append(m.chatHistory, Message{Role: "system", Content: "Error: " + msg.Err.Error()})
			m.refreshViewport()
			m.isStreaming = false
			return m, nil
		}

		if msg.Done {
			aiResponse := Message{Role: "assistant", Content: m.currentBuffer.String()}
			m.chatHistory = append(m.chatHistory, aiResponse)
			saveMessageObject(aiResponse)
			m.currentBuffer.Reset()
			m.isStreaming = false
			m.refreshViewport()
			return m, nil
		}

		m.currentBuffer.WriteString(msg.Content)

		// Render live buffer dynamically
		displayLines := m.renderDisplayLines()
		displayLines = append(displayLines, "AI: "+m.currentBuffer.String())
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

			if strings.HasPrefix(input, "/quit") {
				return m, tea.Quit
			}
			if strings.HasPrefix(input, "/model ") {
				m.modelName = strings.TrimSpace(strings.TrimPrefix(input, "/model "))
				m.textarea.Reset()
				return m, nil
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
				return m, nil
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
				return m, nil
			}
			if strings.HasPrefix(input, "/compress") {
				m.textarea.Reset()
				return m, summarizeCmd(m.modelName, m.chatHistory)
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
			return m, waitForStream(m.streamChan)
		}
	}

	return m, tea.Batch(tiCmd, vpCmd)
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

func main() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
