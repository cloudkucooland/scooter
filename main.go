package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

func getOllamaHost() string {
	if host := os.Getenv("OLLAMA_HOST"); host != "" {
		return strings.TrimRight(host, "/")
	}
	return "http://localhost:11434"
}

func getCacheFile() (string, error) {
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("user cache dir error: %w", err)
	}
	agentDir := filepath.Join(cacheDir, "agent")
	if err := os.MkdirAll(agentDir, 0755); err != nil {
		return "", fmt.Errorf("mkdir error: %w", err)
	}
	return filepath.Join(agentDir, "session.jsonl"), nil
}

func clearCache() {
	cacheFile, err := getCacheFile()
	if err != nil {
		return
	}
	os.Remove(cacheFile)
}

func saveMessageObject(msg Message) error {
	cacheFile, err := getCacheFile()
	if err != nil {
		return err
	}
	f, err := os.OpenFile(cacheFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	if _, err := f.Write(append(data, '\n')); err != nil {
		return err
	}
	return f.Sync()
}

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
	decoder := json.NewDecoder(f)
	for decoder.More() {
		var msg Message
		if err := decoder.Decode(&msg); err == nil {
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
	toolDesc.WriteString("\nIMPORTANT: Only use tools when strictly necessary. For standard conversational responses, answer directly in plain text.\n")
	toolInfo := toolDesc.String()

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
	sb.WriteString("Assistant: ")

	reqBody := map[string]interface{}{
		"model":  modelName,
		"prompt": sb.String(),
		"stream": true,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		sendStreamMsg(ctx, channel, StreamMessage{Err: err})
		return
	}

	req, err := http.NewRequestWithContext(ctx, "POST", fullURL, bytes.NewBuffer(jsonData))
	if err != nil {
		sendStreamMsg(ctx, channel, StreamMessage{Err: err})
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		sendStreamMsg(ctx, channel, StreamMessage{Err: err})
		return
	}
	defer resp.Body.Close()

	rkParser := NewRkToolParser()
	reader := bufio.NewReader(resp.Body)

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err != io.EOF {
				sendStreamMsg(ctx, channel, StreamMessage{Err: err})
			}
			break
		}

		var chunk struct {
			Response string `json:"response"`
			Done     bool   `json:"done"`
		}
		if err := json.Unmarshal(line, &chunk); err == nil {
			cleanText, toolReady, parsedCall := rkParser.ProcessChunk(chunk.Response)

			var calls []ToolCall
			if toolReady {
				calls = append(calls, parsedCall)
			}

			if !sendStreamMsg(ctx, channel, StreamMessage{
				Content:   cleanText,
				ToolCalls: calls,
				Done:      chunk.Done,
			}) {
				return
			}

			if chunk.Done {
				break
			}
		}
	}
}

func sendStreamMsg(ctx context.Context, ch chan<- StreamMessage, msg StreamMessage) bool {
	select {
	case <-ctx.Done():
		return false
	case ch <- msg:
		return true
	}
}

func main() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
