package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

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

func listModelsCmd() tea.Cmd {
	return func() tea.Msg {
		fullURL := getOllamaHost() + "/api/tags"
		resp, err := http.Get(fullURL)
		if err != nil {
			return ModelsMessage{Err: err}
		}
		defer resp.Body.Close()

		var response struct {
			Models []struct {
				Name string `json:"name"`
			} `json:"models"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
			return ModelsMessage{Err: err}
		}

		models := make([]string, len(response.Models))
		for i, m := range response.Models {
			models[i] = m.Name
		}
		return ModelsMessage{Models: models}
	}
}

func statusCmd() tea.Cmd {
	return func() tea.Msg {
		fullURL := getOllamaHost()
		resp, err := http.Get(fullURL)
		if err != nil {
			return StatusMessage{Err: err}
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusOK {
			return StatusMessage{Status: "Ollama server is running (200 OK)"}
		}
		return StatusMessage{Status: fmt.Sprintf("Ollama server returned status: %s", resp.Status)}
	}
}
