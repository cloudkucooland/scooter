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

	tea "github.com/charmbracelet/bubbletea"
)

// Message matches the Ollama JSON wire protocol and is used for session persistence
type Message struct {
	Role    string `json:"role"` // "system", "user", "assistant", "tool"
	Content string `json:"content"`
}

type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
	Stream   bool      `json:"stream"`
}

type ChatResponseChunk struct {
	Message struct {
		Role      string     `json:"role"`
		Content   string     `json:"content"`
		ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	} `json:"message"`
	Done bool `json:"done"`
}

type StreamMessage struct {
	Content   string
	ToolCalls []ToolCall // Threaded here to pass out of goroutine safely
	Done      bool
	Err       error
}

type SummaryMessage struct {
	Summary string
	Err     error
}

type ModelsMessage struct {
	Models []string
	Err    error
}

type StatusMessage struct {
	Status string
	Err    error
}

type ToolCall struct {
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

func getOllamaHost() string {
	if host := os.Getenv("OLLAMA_HOST"); host != "" {
		return strings.TrimRight(host, "/")
	}
	return "http://localhost:11434"
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

func streamOllama(ctx context.Context, modelName string, history []Message, channel chan<- StreamMessage) {
	defer close(channel)

	fullURL := getOllamaHost() + "/api/generate"

	// Prepare tools definition for the system prompt
	tools := getSystemTools()
	var toolDesc strings.Builder
	toolDesc.WriteString("\n\nYou have access to the following tools:\n")
	for _, t := range tools {
		toolDesc.WriteString(fmt.Sprintf("- %s: %s\n", t.Function.Name, t.Function.Description))
		toolDesc.WriteString("  Parameters:\n")
		for pName, pDef := range t.Function.Parameters.Properties {
			toolDesc.WriteString(fmt.Sprintf("    - %s (%s): %s\n", pName, pDef.Type, pDef.Description))
		}
	}
	toolDesc.WriteString("\nIf you need to call a tool, you MUST respond EXACTLY in this format:\n")
	toolDesc.WriteString("<tool_call><function=TOOL_NAME><parameter=PARAM1>VALUE1</parameter><parameter=PARAM2>VALUE2</parameter></tool_call>\n")
	toolDesc.WriteString("Example:\n")
	toolDesc.WriteString("<tool_call><function=write_file><parameter=file_path>/tmp/test.md</parameter><parameter=content>Hello World</parameter></tool_call>\n")
	toolDesc.WriteString("<tool_call><function=get_devices></tool_call>\n")
	toolDesc.WriteString("\nIMPORTANT: Only use tools when strictly necessary to achieve a task (e.g., saving data to a file, checking device status, updating device status). For standard conversational responses, answer directly in plain text without using tool call tags. Do not use tools to document the conversation history.\n")
	toolInfo := toolDesc.String()

	// Construct raw prompt using standard conversational template
	var sb strings.Builder
	for i, msg := range history {
		content := msg.Content
		if i == 0 && msg.Role == "system" {
			content += toolInfo
		}
		
		switch msg.Role {
		case "system":
			sb.WriteString("System: " + content + "\n")
		case "user":
			sb.WriteString("User: " + content + "\n")
		case "assistant":
			sb.WriteString("Assistant: " + content + "\n")
		case "tool":
			sb.WriteString("Tool Result: " + content + "\n")
		}
	}
	sb.WriteString("Assistant: ") // Prompt LLM to continue

	reqBody := map[string]interface{}{
		"model":  modelName,
		"prompt": sb.String(),
		"stream": true,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		channel <- StreamMessage{Err: err}
		return
	}
	logDebug("Sending request to Ollama: %s", string(jsonData))

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

	// Instantiate our custom rkllm structural parser
	rkParser := NewRkToolParser()
	reader := bufio.NewReader(resp.Body)
	
	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			logDebug("Error reading stream: %v", err)
			break
		}
		logDebug("Raw line: %s", string(line))
		
		var chunk struct {
			Response string `json:"response"`
			Done     bool   `json:"done"`
		}
		if err := json.Unmarshal(line, &chunk); err == nil {
			rawText := chunk.Response
			logDebug("Chunk text (raw): %s", rawText)
			
			// Process incoming chunk text through the tag extractor
			cleanText, toolReady, parsedCall := rkParser.ProcessChunk(rawText)

			var calls []ToolCall
			if toolReady {
				calls = append(calls, parsedCall)
			}
			
			channel <- StreamMessage{
				Content:   cleanText,
				ToolCalls: calls, 
				Done:      chunk.Done,
			}
			if chunk.Done {
				break
			}
		}
	}
}

func main() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

