package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"codeberg.org/indievisible/scooter/pkg/knowledge"
	"codeberg.org/indievisible/scooter/pkg/rkllm"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/urfave/cli/v3"
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
	sb.WriteString("## Operational Tool Execution Policy\n")
	sb.WriteString("1. You operate in a live production environment with full real-time tool execution capabilities.\n")
	sb.WriteString("2. When a tool is required to fulfill a user's request, you MUST invoke it immediately. Do NOT explain, discuss, or speculate about tool availability, network access, or limitations in conversational text.\n")
	sb.WriteString("3. NEVER claim a tool failed or is unavailable without attempting the execution first.\n")
	sb.WriteString("4. To invoke a tool, output ONLY the JSON request wrapped in exact `[TOOL_CALL]` and `[/TOOL_CALL]` tags:\n")
	sb.WriteString("   [TOOL_CALL]{\"name\":\"TOOL_NAME\",\"arguments\":{}}[/TOOL_CALL]\n")
	sb.WriteString("5. Arguments must be a valid, single-line JSON object with double-quoted keys and string values.\n\n")

	fmt.Fprintf(&sb, "File and directory operations are restricted to: %s\n\n", workingDir)

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

func main() {
	defer closeLogger()
	defer func() {
		closed := make(chan struct{})
		go func() {
			rkllm.CloseRKLLMSession()
			close(closed)
		}()
		select {
		case <-closed:
		case <-time.After(2 * time.Second):
			logDebug("RKLLM session teardown did not finish within 2 seconds; exiting")
		}
	}()

	cwd, err := os.Getwd()
	if err != nil {
		panic(err)
	}
	workingDir, err = filepath.Abs(cwd)
	if err != nil {
		panic(err)
	}

	rkllm.GetActiveContextLimit = getActiveContextLimit
	rkllm.GetActiveToolset = GetActiveToolset
	rkllm.GetCacheFile = getCacheFile
	rkllm.IsCacheEnabled = func() bool {
		return loadConfig().RKLLMCacheEnabled
	}
	rkllm.BuildSystemPrompt = func() string {
		base := buildToolInstructions(getSystemTools())
		if reminder := FormatTasksReminder(); reminder != "" {
			base += "\n\n" + reminder
		}
		return base
	}

	app := &cli.Command{
		Name:  "scooter",
		Usage: "Local-first AI agent for RK3588",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:    "toolset",
				Aliases: []string{"t"},
				Value:   "",
				Usage:   "Active toolset (dev, web, research, all)",
			},
			&cli.StringFlag{
				Name:    "model",
				Aliases: []string{"m"},
				Usage:   "RKLLM model path or name",
			},
			&cli.StringFlag{
				Name:    "prompt",
				Aliases: []string{"p"},
				Usage:   "Run a single prompt non-interactively and exit",
			},
			&cli.StringFlag{
				Name:    "ingest",
				Aliases: []string{"i"},
				Usage:   "Ingest a file or glob pattern directly into knowledge base",
			},
			&cli.StringFlag{
				Name:    "collection",
				Aliases: []string{"c"},
				Usage:   "Collection name for direct ingestion (defaults to 'main')",
			},
			&cli.BoolFlag{
				Name:    "debug",
				Aliases: []string{"d"},
				Usage:   "Enable RKNN and RKLLM debug logging output",
			},
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			if cmd.Bool("debug") {
				_ = os.Setenv("RKNN_LOG_LEVEL", "1")
				_ = os.Setenv("RKLLM_LOG_LEVEL", "1")
			} else {
				_ = os.Setenv("RKNN_LOG_LEVEL", "0")
				_ = os.Setenv("RKLLM_LOG_LEVEL", "0")
			}

			toolset := resolveToolset(cmd.String("toolset"))
			if err := SetActiveToolset(toolset); err != nil {
				return err
			}

			ingestPath := cmd.String("ingest")
			if ingestPath != "" {
				collection := cmd.String("collection")
				if collection == "" {
					collection = "main"
				}
				return runDirectIngest(ingestPath, cmd.Args().Slice(), collection)
			}

			modelName := cmd.String("model")
			if modelName == "" {
				modelName = loadConfig().DefaultModel
			}

			promptText := cmd.String("prompt")
			if promptText != "" {
				return runNonInteractive(modelName, toolset, promptText)
			}

			p := tea.NewProgram(initialModel(), tea.WithAltScreen())
			if _, err := p.Run(); err != nil {
				return err
			}
			return nil
		},
	}

	if err := app.Run(context.Background(), os.Args); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func runNonInteractive(modelName, toolset, promptText string) error {
	if err := SetActiveToolset(resolveToolset(toolset)); err != nil {
		return err
	}
	resolvedModel := rkllm.GetRKLLMModelPath(modelName)
	detected := detectContextLength(resolvedModel)
	setActiveContextLimit(detected)

	var history []Message
	history = append(history, Message{Role: "user", Content: promptText})

	for rounds := 0; rounds < 5; rounds++ {
		ch := make(chan StreamMessage, 10)
		ctx := context.Background()
		go rkllm.StreamRKLLM(ctx, resolvedModel, history, ch)

		var assistantContent strings.Builder
		var toolCalls []ToolCall
		done := false

		for !done {
			msg, ok := <-ch
			if !ok {
				break
			}
			if msg.Err != nil {
				return msg.Err
			}
			if msg.Content != "" {
				fmt.Print(msg.Content)
				assistantContent.WriteString(msg.Content)
			}
			if len(msg.ToolCalls) > 0 {
				toolCalls = msg.ToolCalls
			}
			if msg.Done {
				done = true
			}
		}
		fmt.Println()

		assistantMsg := Message{
			Role:      "assistant",
			Content:   assistantContent.String(),
			ToolCalls: toolCalls,
		}
		history = append(history, assistantMsg)

		if len(toolCalls) == 0 {
			break
		}

		for _, call := range toolCalls {
			fmt.Fprintf(os.Stderr, "[Executing tool: %s]\n", call.Function.Name)
			result := dispatchToolCall(call.Function.Name, call.Function.Arguments)
			toolMsg := Message{
				Role:     "tool",
				ToolName: call.Function.Name,
				Content:  result,
			}
			history = append(history, toolMsg)
			fmt.Fprintf(os.Stderr, "[Tool result: %s]\n", result)
		}
	}
	return nil
}

func runDirectIngest(ingestVal string, extraArgs []string, collection string) error {
	var patterns []string
	if ingestVal != "" {
		patterns = append(patterns, ingestVal)
	}
	patterns = append(patterns, extraArgs...)

	baseDir, err := os.Getwd()
	if err != nil {
		return err
	}

	var allFiles []string
	seen := make(map[string]bool)

	for _, pat := range patterns {
		fullPattern := pat
		if !filepath.IsAbs(fullPattern) {
			fullPattern = filepath.Join(baseDir, fullPattern)
		}
		matches, err := filepath.Glob(fullPattern)
		if err != nil || len(matches) == 0 {
			literal := filepath.Join(baseDir, pat)
			if _, statErr := os.Stat(literal); statErr == nil {
				matches = []string{literal}
			} else {
				matches = []string{pat}
			}
		}

		for _, m := range matches {
			if !seen[m] {
				seen[m] = true
				allFiles = append(allFiles, m)
			}
		}
	}

	if len(allFiles) == 0 {
		return fmt.Errorf("no files found matching ingest pattern(s)")
	}

	kdb, err := knowledge.Open()
	if err != nil {
		return fmt.Errorf("open knowledge base: %w", err)
	}
	defer kdb.Close()

	count := 0
	var errors []string
	for _, file := range allFiles {
		absPath, valErr := validatePath(file)
		if valErr != nil {
			errors = append(errors, fmt.Sprintf("%s: %v", filepath.Base(file), valErr))
			continue
		}
		docID, ingestErr := kdb.IngestFile(absPath, collection)
		if ingestErr != nil {
			errors = append(errors, fmt.Sprintf("%s: %v", filepath.Base(file), ingestErr))
		} else {
			fmt.Printf("Successfully ingested %s into collection %q (Doc ID: %s)\n", filepath.Base(file), collection, docID)
			count++
		}
	}

	if count == 0 && len(errors) > 0 {
		return fmt.Errorf("ingest failed: %s", strings.Join(errors, "; "))
	}
	if len(errors) > 0 {
		fmt.Printf("Completed with %d success(es) and %d error(s).\n", count, len(errors))
	} else {
		fmt.Printf("Successfully ingested %d file(s) into collection %q.\n", count, collection)
	}
	return nil
}
