package main

import (
	"context"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

func compressHistoryCmd(modelName string, history []Message, force bool) tea.Cmd {
	source, retained := planCompression(history, force)
	if len(source) == 0 {
		return func() tea.Msg {
			return SummaryMessage{Retained: retained}
		}
	}
	return summarizeCmd(modelName, source, retained)
}

func summarizeCmd(modelName string, messages, retained []Message) tea.Cmd {
	return func() tea.Msg {
		_, cancel := context.WithTimeout(context.Background(), 60*time.Minute)
		defer cancel()

		summaryText, err := runRKLLMSummary(modelName, messages)
		if err != nil {
			return SummaryMessage{Err: err}
		}
		return SummaryMessage{
			Summary:  truncateToTokenBudget(summaryText, SummaryReserveTokens),
			Retained: retained,
		}
	}
}

func listModelsCmd() tea.Cmd {
	return func() tea.Msg {
		models := listRKLLMModels()
		return ModelsMessage{Models: models}
	}
}

func statusCmd() tea.Cmd {
	return func() tea.Msg {
		status := checkRKLLMStatus()
		return StatusMessage{Status: status}
	}
}
