package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

var workingDir string

func getCacheFile() (string, error) {
	if custom := os.Getenv("SCOOTER_CACHE_FILE"); custom != "" {
		dir := filepath.Dir(custom)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return "", fmt.Errorf("mkdir error: %w", err)
		}
		return custom, nil
	}

	cacheDir, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("user cache dir error: %w", err)
	}

	scooterDir := filepath.Join(cacheDir, "scooter")
	if customDir := os.Getenv("SCOOTER_CACHE_DIR"); customDir != "" {
		scooterDir = customDir
	}
	if err := os.MkdirAll(scooterDir, 0755); err != nil {
		return "", fmt.Errorf("mkdir error: %w", err)
	}

	targetDir := workingDir
	if targetDir == "" {
		if cwd, err := os.Getwd(); err == nil {
			targetDir, _ = filepath.Abs(cwd)
		}
	}

	hash := sha256.Sum256([]byte(targetDir))
	hashHex := fmt.Sprintf("%x", hash)[:12]
	base := filepath.Base(targetDir)
	if base == "" || base == "." || base == "/" {
		base = "root"
	}
	var cleanBase strings.Builder
	for _, r := range base {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			cleanBase.WriteRune(r)
		}
	}
	slug := cleanBase.String()
	if slug == "" {
		slug = "workspace"
	}

	return filepath.Join(scooterDir, fmt.Sprintf("%s-%s.jsonl", slug, hashHex)), nil
}

func clearCache() {
	cacheFile, err := getCacheFile()
	if err != nil {
		logDebug("cache clear failed: %v", err)
		return
	}
	if err := os.Remove(cacheFile); err != nil && !os.IsNotExist(err) {
		logDebug("cache clear failed for %s: %v", cacheFile, err)
	}
}

func saveMessageObject(msg Message) error {
	cacheFile, err := getCacheFile()
	if err != nil {
		logDebug("cache save failed: %v", err)
		return err
	}
	f, err := os.OpenFile(cacheFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		logDebug("cache save failed opening %s: %v", cacheFile, err)
		return err
	}
	defer f.Close()

	data, err := json.Marshal(msg)
	if err != nil {
		logDebug("cache save failed encoding message: %v", err)
		return err
	}
	if _, err := f.Write(append(data, '\n')); err != nil {
		logDebug("cache save failed writing %s: %v", cacheFile, err)
		return err
	}
	if err := f.Sync(); err != nil {
		logDebug("cache save failed syncing %s: %v", cacheFile, err)
		return err
	}
	return nil
}

func loadHistoryObjects() []Message {
	cacheFile, err := getCacheFile()
	if err != nil {
		logDebug("cache load failed: %v", err)
		return nil
	}
	f, err := os.Open(cacheFile)
	if err != nil {
		if !os.IsNotExist(err) {
			logDebug("cache load failed opening %s: %v", cacheFile, err)
		}
		return nil
	}
	defer f.Close()

	var history []Message
	decoder := json.NewDecoder(f)
	for decoder.More() {
		var msg Message
		if err := decoder.Decode(&msg); err == nil {
			if msg.Role == "system" && strings.HasPrefix(msg.Content, "Context loaded from ") {
				msg.Content = truncateToTokenBudget(msg.Content, getHydrationTokenLimit())
			}
			history = append(history, msg)
		} else {
			logDebug("cache load skipped malformed record in %s: %v", cacheFile, err)
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
			boundedContent := truncateToTokenBudget(string(content), getHydrationTokenLimit())
			return &Message{
				Role:    "system",
				Content: fmt.Sprintf("Context loaded from %s:\n%s", file, boundedContent),
			}
		}
	}
	return nil
}

func buildToolInstructions(tools []Tool) string {
	var sb strings.Builder
	sb.WriteString("You can call available local tools when necessary. Do not claim that a tool was run unless a Tool result is present.\n")
	sb.WriteString("CRITICAL: When a tool is needed, you MUST output the `[TOOL_CALL]` block to execute it. Do NOT merely describe, discuss, or reason about calling a tool in plain text without outputting the `[TOOL_CALL]` markers.\n")
	sb.WriteString("For a tool call, output only this exact JSON object between the markers, with no Markdown or commentary inside the markers:\n")
	sb.WriteString("[TOOL_CALL]{\"name\":\"TOOL_NAME\",\"arguments\":{}}[/TOOL_CALL]\n")
	sb.WriteString("Use valid JSON: double-quote keys and string values; arguments must be an object. A response may contain multiple complete tool calls. If reasoning or thinking through steps, always conclude your response with the required `[TOOL_CALL]` block to invoke the tool. Otherwise, answer in plain text.\n\n")
	fmt.Fprintf(&sb, "File and directory operations are restricted to the working directory: %s (paths may be relative or absolute within this directory).\n\n", workingDir)
	sb.WriteString("Available tools:\n")

	sortedTools := append([]Tool(nil), tools...)
	sort.Slice(sortedTools, func(i, j int) bool {
		return sortedTools[i].Function.Name < sortedTools[j].Function.Name
	})
	for _, tool := range sortedTools {
		sb.WriteString(fmt.Sprintf("- %s: %s\n", tool.Function.Name, tool.Function.Description))
		if len(tool.Function.Parameters.Properties) == 0 {
			sb.WriteString("  arguments: {}\n")
			continue
		}

		parameterNames := make([]string, 0, len(tool.Function.Parameters.Properties))
		for name := range tool.Function.Parameters.Properties {
			parameterNames = append(parameterNames, name)
		}
		sort.Strings(parameterNames)
		for _, name := range parameterNames {
			parameter := tool.Function.Parameters.Properties[name]
			required := "optional"
			if contains(tool.Function.Parameters.Required, name) {
				required = "required"
			}
			sb.WriteString(fmt.Sprintf("  - %s (%s, %s): %s\n", name, parameter.Type, required, parameter.Description))
		}
	}
	return sb.String()
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

/* func sendStreamMsg(ctx context.Context, ch chan<- StreamMessage, msg StreamMessage) bool {
	select {
	case <-ctx.Done():
		return false
	case ch <- msg:
		return true
	}
} */

func main() {
	defer closeLogger()
	defer closeRKLLMSession()

	cwd, err := os.Getwd()
	if err != nil {
		panic(err)
	}
	workingDir, err = filepath.Abs(cwd)
	if err != nil {
		panic(err)
	}

	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
