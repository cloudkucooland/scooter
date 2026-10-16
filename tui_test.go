package main

import (
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
)

func TestUpdateCtrlLRedrawsTUI(t *testing.T) {
	m := model{
		textarea:      textarea.New(),
		viewport:      viewport.New(80, 24),
		currentBuffer: &strings.Builder{},
	}

	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlL})
	if cmd == nil {
		t.Fatal("expected non-nil cmd on KeyCtrlL (tea.ClearScreen)")
	}
}

func TestUpdateEscStopsStreaming(t *testing.T) {
	canceled := false
	m := model{
		textarea:      textarea.New(),
		viewport:      viewport.New(80, 24),
		isStreaming:   true,
		currentBuffer: &strings.Builder{},
		streamCancel: func() {
			canceled = true
		},
	}
	m.currentBuffer.WriteString("partial output")

	updatedM, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if cmd != nil {
		t.Errorf("expected nil cmd on Esc, got %v", cmd)
	}

	resModel, ok := updatedM.(model)
	if !ok {
		t.Fatalf("expected updated model to be model, got %T", updatedM)
	}

	if resModel.isStreaming {
		t.Errorf("expected isStreaming to be false after Esc")
	}
	if !canceled {
		t.Errorf("expected streamCancel func to be called")
	}

	lastMsg := resModel.chatHistory[len(resModel.chatHistory)-1]
	if !strings.Contains(lastMsg.Content, "[aborted]") {
		t.Errorf("expected last message to contain [aborted], got %q", lastMsg.Content)
	}
}
