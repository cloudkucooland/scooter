package main

import (
	"encoding/json"
	"os"
	"path/filepath"
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
	if cmd == nil {
		t.Errorf("expected non-nil cmd on Esc with partial output, got nil")
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

func TestModelAutocompletePopup(t *testing.T) {
	m := model{
		textarea: textarea.New(),
	}
	m.textarea.SetValue("/model")
	popup := m.renderAutocompletePopup()
	if !strings.Contains(popup, "RKLLM Models") {
		t.Errorf("expected popup to contain 'RKLLM Models', got %q", popup)
	}
}

func TestShellCommandConfirmationPrompt(t *testing.T) {
	m := model{
		textarea:      textarea.New(),
		viewport:      viewport.New(80, 24),
		currentBuffer: &strings.Builder{},
		confirmingTool: &ToolCall{
			Function: struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}{
				Name:      "shell",
				Arguments: []byte(`{"command":"echo hello"}`),
			},
		},
	}

	// Pressing 'n' should cancel execution
	updatedM, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'n'}})
	resModel, ok := updatedM.(model)
	if !ok {
		t.Fatalf("expected model, got %T", updatedM)
	}
	if resModel.confirmingTool != nil {
		t.Fatal("expected confirmingTool to be cleared after cancellation")
	}
	lastMsg := resModel.chatHistory[len(resModel.chatHistory)-1]
	if !strings.Contains(lastMsg.Content, "cancelled") {
		t.Fatalf("expected cancellation message, got %q", lastMsg.Content)
	}
}

func TestBangCommandExecution(t *testing.T) {
	m := model{
		textarea:      textarea.New(),
		viewport:      viewport.New(80, 24),
		currentBuffer: &strings.Builder{},
	}

	cmd := m.handleCommand("!echo hello_bang")
	if cmd != nil {
		t.Errorf("expected nil cmd from synchronous bang command, got %v", cmd)
	}

	if len(m.chatHistory) < 2 {
		t.Fatalf("expected at least 2 chat history messages (user prompt and tool result), got %d", len(m.chatHistory))
	}

	userMsg := m.chatHistory[len(m.chatHistory)-2]
	if userMsg.Role != "user" || userMsg.Content != "!echo hello_bang" {
		t.Errorf("expected user message '!echo hello_bang', got %+v", userMsg)
	}

	toolMsg := m.chatHistory[len(m.chatHistory)-1]
	if toolMsg.Role != "tool" || toolMsg.ToolName != "shell" || !strings.Contains(toolMsg.Content, "hello_bang") {
		t.Errorf("expected tool message with output containing 'hello_bang', got %+v", toolMsg)
	}
}

func TestDisableAndEnableCacheCommands(t *testing.T) {
	m := model{
		textarea:      textarea.New(),
		viewport:      viewport.New(80, 24),
		currentBuffer: &strings.Builder{},
	}

	cmd1 := m.handleCommand("/disablecache")
	if cmd1 != nil {
		t.Errorf("expected nil cmd for /disablecache, got %v", cmd1)
	}
	if loadConfig().RKLLMCacheEnabled {
		t.Errorf("expected RKLLMCacheEnabled to be false after /disablecache")
	}

	cmd2 := m.handleCommand("/enablecache")
	if cmd2 != nil {
		t.Errorf("expected nil cmd for /enablecache, got %v", cmd2)
	}
	if !loadConfig().RKLLMCacheEnabled {
		t.Errorf("expected RKLLMCacheEnabled to be true after /enablecache")
	}
}

func TestStatusBarCacheIndicator(t *testing.T) {
	m := model{
		textarea:      textarea.New(),
		viewport:      viewport.New(80, 24),
		modelName:     "test_model",
		currentBuffer: &strings.Builder{},
	}

	cfg := loadConfig()
	cfg.RKLLMCacheEnabled = true
	_ = saveConfig(cfg)
	statusBar1 := m.renderStatusBar()
	if !strings.Contains(statusBar1, "Cache: ⚡") {
		t.Errorf("expected status bar to contain 'Cache: ⚡', got %q", statusBar1)
	}

	cfg.RKLLMCacheEnabled = false
	_ = saveConfig(cfg)
	statusBar2 := m.renderStatusBar()
	if !strings.Contains(statusBar2, "Cache: off") {
		t.Errorf("expected status bar to contain 'Cache: off', got %q", statusBar2)
	}

	// restore
	cfg.RKLLMCacheEnabled = true
	_ = saveConfig(cfg)
}

func TestCacheCommandsInAvailableCommands(t *testing.T) {
	foundEnable := false
	foundDisable := false
	for _, c := range availableCommands {
		if c.cmd == "/enablecache" {
			foundEnable = true
		}
		if c.cmd == "/disablecache" {
			foundDisable = true
		}
	}
	if !foundEnable {
		t.Errorf("expected /enablecache in availableCommands")
	}
	if !foundDisable {
		t.Errorf("expected /disablecache in availableCommands")
	}
}

func TestImagineCommandInAvailableCommands(t *testing.T) {
	foundImagine := false
	for _, c := range availableCommands {
		if c.cmd == "/imagine" {
			foundImagine = true
		}
	}
	if !foundImagine {
		t.Errorf("expected /imagine in availableCommands")
	}
}

func TestDescribeCommandInAvailableCommands(t *testing.T) {
	foundDescribe := false
	for _, c := range availableCommands {
		if c.cmd == "/describe" {
			foundDescribe = true
		}
	}
	if !foundDescribe {
		t.Errorf("expected /describe in availableCommands")
	}
}

func TestParseImagineCommandArgs(t *testing.T) {
	prompt, outPath, steps, seed, guidance, negative, err := parseImagineCommandArgs(`a red scooter --out test.png --steps 7 --seed 42 --guidance 1.8 --negative "extra wheels"`)
	if err != nil {
		t.Fatal(err)
	}
	if prompt != "a red scooter" {
		t.Errorf("expected prompt 'a red scooter', got %q", prompt)
	}
	if outPath != "test.png" {
		t.Errorf("expected outPath 'test.png', got %q", outPath)
	}
	if guidance != 1.8 {
		t.Errorf("expected guidance 1.8, got %f", guidance)
	}
	if steps != 7 || seed != 42 || negative != "extra wheels" {
		t.Errorf("expected steps=7 seed=42 negative='extra wheels', got steps=%d seed=%d negative=%q", steps, seed, negative)
	}
}

func TestParseImagineNegativePromptAndPromptOrdering(t *testing.T) {
	prompt, _, _, _, _, negative, err := parseImagineCommandArgs(`--negative "no text, no watermark" a red scooter --guidance 1.5`)
	if err != nil {
		t.Fatal(err)
	}
	if prompt != "a red scooter" || negative != "no text, no watermark" {
		t.Fatalf("got prompt=%q negative=%q", prompt, negative)
	}
	prompt, _, _, _, _, negative, err = parseImagineCommandArgs(`--negative none a red scooter`)
	if err != nil {
		t.Fatal(err)
	}
	if prompt != "a red scooter" || negative != "none" {
		t.Fatalf("got prompt=%q negative=%q", prompt, negative)
	}

	prompt, _, _, _, _, negative, err = parseImagineCommandArgs(`-n "no logos, no text" a blue scooter`)
	if err != nil {
		t.Fatal(err)
	}
	if prompt != "a blue scooter" || negative != "no logos, no text" {
		t.Fatalf("short alias got prompt=%q negative=%q", prompt, negative)
	}
}

func TestParseImagineRejectsInvalidOptions(t *testing.T) {
	tests := []struct {
		name string
		args string
	}{
		{name: "unterminated negative quote", args: `a red scooter --negative "no text`},
		{name: "missing negative value", args: `a red scooter --negative`},
		{name: "missing output value", args: `a red scooter --out`},
		{name: "invalid steps", args: `a red scooter --steps many`},
		{name: "invalid seed", args: `a red scooter --seed many`},
		{name: "invalid guidance", args: `a red scooter --guidance many`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, _, _, _, _, _, err := parseImagineCommandArgs(tt.args); err == nil {
				t.Fatalf("expected error parsing %q", tt.args)
			}
		})
	}
}

func TestRKNNCommandsInAvailableCommands(t *testing.T) {
	foundModels := false
	foundModel := false
	for _, c := range availableCommands {
		if c.cmd == "/rknnmodels" {
			foundModels = true
		}
		if c.cmd == "/rknnmodel" {
			foundModel = true
		}
	}
	if !foundModels {
		t.Errorf("expected /rknnmodels in availableCommands")
	}
	if !foundModel {
		t.Errorf("expected /rknnmodel in availableCommands")
	}
}

func TestRKNNModelSwitchCommand(t *testing.T) {
	modelDir := t.TempDir()
	modelPath := filepath.Join(modelDir, "test-custom-rknn")
	if err := os.MkdirAll(filepath.Join(modelPath, "unet"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelPath, "unet", "model.rknn"), []byte("test"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("SCOOTER_RKNN_MODELS_DIR", modelDir)
	t.Setenv("RKNN_MODELS_DIR", "")
	t.Setenv("RKNN_MODEL_DIR", "")
	t.Setenv("SCOOTER_RKNN_MODEL", "")
	t.Setenv("RKNN_MODEL", "")
	m := model{
		textarea:      textarea.New(),
		viewport:      viewport.New(80, 24),
		currentBuffer: &strings.Builder{},
	}

	cmd := m.handleCommand("/rknnmodel " + modelPath)
	if cmd != nil {
		t.Errorf("expected nil cmd for switching model, got %v", cmd)
	}
	if m.rknnModel != modelPath {
		t.Errorf("expected m.rknnModel %q, got %q", modelPath, m.rknnModel)
	}
	if getActiveRKNNModel() != m.rknnModel {
		t.Errorf("expected active RKNN model %q, got %q", m.rknnModel, getActiveRKNNModel())
	}
	if len(m.chatHistory) == 0 {
		t.Fatalf("expected system message in chatHistory after switching model")
	}
	lastMsg := m.chatHistory[len(m.chatHistory)-1]
	if !strings.Contains(lastMsg.Content, "Switched RKNN model to "+modelPath) {
		t.Errorf("unexpected system message content: %q", lastMsg.Content)
	}

	// Test /rknnmodels command returns a tea.Cmd
	cmdModels := m.handleCommand("/rknnmodels")
	if cmdModels == nil {
		t.Errorf("expected non-nil cmd for /rknnmodels")
	}
}

func TestRKNNModelAutocomplete(t *testing.T) {
	m := model{
		textarea: textarea.New(),
		viewport: viewport.New(80, 24),
	}

	m.textarea.SetValue("/rknnmodel")
	popup := m.renderAutocompletePopup()
	if !strings.Contains(popup, "RKNN Models") {
		t.Errorf("expected popup to contain 'RKNN Models', got %q", popup)
	}
}

func TestVisionModelCommandsInAvailableCommands(t *testing.T) {
	foundModels := false
	foundModel := false
	for _, c := range availableCommands {
		if c.cmd == "/visionmodels" {
			foundModels = true
		}
		if c.cmd == "/visionmodel" {
			foundModel = true
		}
	}
	if !foundModels {
		t.Errorf("expected /visionmodels in availableCommands")
	}
	if !foundModel {
		t.Errorf("expected /visionmodel in availableCommands")
	}
}

func TestVisionModelSwitchCommand(t *testing.T) {
	modelDir := t.TempDir()
	modelPath := filepath.Join(modelDir, "test-vision-model.rknn")
	if err := os.WriteFile(modelPath, []byte("test"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("SCOOTER_RKNN_VISION_MODEL", "")
	t.Setenv("RKNN_VISION_MODEL", "")
	t.Setenv("SCOOTER_RKNN_MODELS_DIR", modelDir)
	t.Setenv("RKNN_MODELS_DIR", "")
	t.Setenv("RKNN_MODEL_DIR", "")
	m := model{
		textarea:      textarea.New(),
		viewport:      viewport.New(80, 24),
		currentBuffer: &strings.Builder{},
	}

	cmd := m.handleCommand("/visionmodel " + modelPath)
	if cmd != nil {
		t.Errorf("expected nil cmd for switching vision model, got %v", cmd)
	}
	if m.visionModel != modelPath {
		t.Errorf("expected m.visionModel %q, got %q", modelPath, m.visionModel)
	}
	if len(m.chatHistory) == 0 {
		t.Fatalf("expected system message in chatHistory after switching vision model")
	}
	lastMsg := m.chatHistory[len(m.chatHistory)-1]
	if !strings.Contains(lastMsg.Content, "Switched vision model to") {
		t.Errorf("unexpected system message content: %q", lastMsg.Content)
	}
}

func TestVisionModelBareCommandListsModels(t *testing.T) {
	m := model{
		textarea:      textarea.New(),
		viewport:      viewport.New(80, 24),
		currentBuffer: &strings.Builder{},
	}

	cmd := m.handleCommand("/visionmodel")
	if cmd == nil {
		t.Errorf("expected non-nil cmd for /visionmodel (bare), should list vision models")
	}
}

func TestVisionModelsCommandListsModels(t *testing.T) {
	m := model{
		textarea:      textarea.New(),
		viewport:      viewport.New(80, 24),
		currentBuffer: &strings.Builder{},
	}

	cmd := m.handleCommand("/visionmodels")
	if cmd == nil {
		t.Errorf("expected non-nil cmd for /visionmodels")
	}
}

func TestVisionModelAutocompletePopup(t *testing.T) {
	m := model{
		textarea: textarea.New(),
		viewport: viewport.New(80, 24),
	}

	m.textarea.SetValue("/visionmodel")
	popup := m.renderAutocompletePopup()
	if !strings.Contains(popup, "Vision Models") {
		t.Errorf("expected popup to contain 'Vision Models', got %q", popup)
	}
}
