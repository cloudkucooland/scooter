package main

import (
	"context"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

func compressHistoryCmd(modelName string, history []Message, force bool, generation uint64) tea.Cmd {
	source, retained := planCompression(history, force)
	if len(source) == 0 {
		return func() tea.Msg {
			return SummaryMessage{Retained: retained, Generation: generation}
		}
	}
	return summarizeCmd(modelName, source, retained, generation)
}

func summarizeCmd(modelName string, messages, retained []Message, generation uint64) tea.Cmd {
	return func() tea.Msg {
		_, cancel := context.WithTimeout(context.Background(), 60*time.Minute)
		defer cancel()

		summaryText, err := runRKLLMSummary(modelName, messages)
		if err != nil {
			return SummaryMessage{Err: err, Generation: generation}
		}
		return SummaryMessage{
			Summary:    truncateToTokenBudget(summaryText, getSummaryReserveTokens()),
			Retained:   retained,
			Generation: generation,
		}
	}
}

func listModelsCmd() tea.Cmd {
	return func() tea.Msg {
		models := listRKLLMModels()
		return ModelsMessage{Models: models}
	}
}

func listToolsetsCmd() tea.Cmd {
	return func() tea.Msg {
		toolsets := listToolsetsNames()
		return ToolsetsMessage{Toolsets: toolsets}
	}
}

func statusCmd(modelName string) tea.Cmd {
	return func() tea.Msg {
		status := checkRKLLMStatus(modelName)
		return StatusMessage{Status: status}
	}
}
