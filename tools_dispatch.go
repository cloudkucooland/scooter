package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"google.golang.org/genai"
)

func dispatchToolCall(name string, rawArgs json.RawMessage) string {
	logDebug("dispatchToolCall called: %s, Args: %s", name, string(rawArgs))
	if strings.TrimSpace(name) == "" {
		return "System Error: tool name is required"
	}
	switch name {
	case "_error":
		var errMsg string
		if err := json.Unmarshal(rawArgs, &errMsg); err != nil {
			errMsg = string(rawArgs)
		}
		return fmt.Sprintf("Tool Error: %s", errMsg)

	case "apply_patch":
		var args ApplyPatchArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		return applyPatch(args.Patch)

	case "file_metadata":
		var args FileMetadataArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		if strings.TrimSpace(args.FilePath) == "" {
			return "System Error: file_path is required"
		}
		absPath, err := validatePath(args.FilePath)
		if err != nil {
			return fmt.Sprintf("System Error: %v", err)
		}
		return fileMetadata(absPath)

	case "edit_file":
		var args EditFileArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		if strings.TrimSpace(args.FilePath) == "" {
			return "System Error: file_path is required"
		}
		if len(args.Operations) == 0 {
			return "System Error: operations are required"
		}
		absPath, err := validatePath(args.FilePath)
		if err != nil {
			return fmt.Sprintf("System Error: %v", err)
		}
		res, err := editFile(absPath, args.Operations, args.PreviewOnly)
		if err != nil {
			return fmt.Sprintf("System Error: %v", err)
		}
		return res

	case "outsource":
		var args OutsourceArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		if strings.TrimSpace(args.Prompt) == "" {
			return "System Error: prompt is required"
		}

		apiKey := getGoogleAPIKey()
		if apiKey == "" {
			return "System Error: Google API key is not configured. Set GOOGLE_API_KEY environment variable or 'google_api_key' in your config file (~/.config/scooter/config.json)."
		}

		modelName := getGoogleGenAIModel()
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		client, err := genai.NewClient(ctx, &genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		})
		if err != nil {
			return fmt.Sprintf("System Error creating GenAI client: %v", err)
		}

		result, err := client.Models.GenerateContent(ctx, modelName, genai.Text(args.Prompt), nil)
		if err != nil {
			return fmt.Sprintf("System Error generating content from Google GenAI: %v", err)
		}

		if result == nil || len(result.Candidates) == 0 {
			return "System Error: No response candidates returned from Google GenAI."
		}

		return result.Text()

	case "web_fetch":
		var args WebFetchArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		targetPrompt := args.Prompt
		if targetPrompt == "" && args.URL != "" {
			targetPrompt = args.URL
		}

		return webFetch(targetPrompt)

	case "git_status", "git_diff":
		var args GitPathArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		if name == "git_status" {
			return runGit("status", args.Path)
		}
		return runGit("diff", args.Path)

	case "find_files":
		var args FindFilesArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		if strings.TrimSpace(args.Pattern) == "" {
			return "System Error: pattern is required"
		}
		if args.MaxResults < 0 {
			return "System Error: max_results must not be negative"
		}
		maxResults := args.MaxResults
		if maxResults == 0 {
			maxResults = 100
		}
		baseDir := workingDir
		if baseDir == "" {
			if cwd, err := os.Getwd(); err == nil {
				baseDir, _ = filepath.Abs(cwd)
			}
		}
		result, err := findFileMatches(baseDir, filepath.ToSlash(args.Pattern), maxResults, 8192)
		if err != nil {
			return fmt.Sprintf("System Error finding files: %v", err)
		}
		if result == "" {
			return "No matching files found."
		}
		return result

	case "replace_text":
		var args ReplaceTextArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		if strings.TrimSpace(args.FilePath) == "" {
			return "System Error: file_path is required"
		}
		if args.OldText == "" {
			return "System Error: old_text is required"
		}
		if args.ExpectedMatches < 0 {
			return "System Error: expected_matches must not be negative"
		}
		expected := args.ExpectedMatches
		if expected == 0 {
			expected = 1
		}
		absPath, err := validatePath(args.FilePath)
		if err != nil {
			return fmt.Sprintf("System Error: %v", err)
		}
		content, err := os.ReadFile(absPath)
		if err != nil {
			return fmt.Sprintf("System Error reading path: %v", err)
		}
		matches := strings.Count(string(content), args.OldText)
		if matches != expected {
			return fmt.Sprintf("System Error: found %d matches for old_text, expected %d; file was not changed", matches, expected)
		}
		updated := strings.Replace(string(content), args.OldText, args.NewText, expected)
		if err := os.WriteFile(absPath, []byte(updated), 0644); err != nil {
			return fmt.Sprintf("System Error writing path: %v", err)
		}
		return fmt.Sprintf("Success: replaced %d match(es) in %s", matches, absPath)

	case "shell":
		var args ShellArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		return runShell(args.Command)

	case "write_file":
		var args WriteFileArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		if strings.TrimSpace(args.FilePath) == "" {
			return "System Error: file_path is required"
		}

		absPath, err := validatePath(args.FilePath)
		if err != nil {
			return fmt.Sprintf("System Error: %v", err)
		}

		dir := filepath.Dir(absPath)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Sprintf("System Error creating directory %s: %v", dir, err)
		}

		if err := os.WriteFile(absPath, []byte(args.Content), 0644); err != nil {
			return fmt.Sprintf("System Error writing path: %v", err)
		}
		res := fmt.Sprintf("Success: Intercepted text stream and wrote file directly to %s", absPath)
		logDebug("dispatchToolCall success: %s", res)
		return res

	case "read_file":
		var args ReadFileArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		if strings.TrimSpace(args.FilePath) == "" {
			return "System Error: file_path is required"
		}
		if args.StartLine < 0 || args.EndLine < 0 ||
			(args.EndLine > 0 && args.StartLine == 0) ||
			(args.StartLine > 0 && args.EndLine > 0 && args.EndLine < args.StartLine) {
			return "System Error: invalid line range; use positive 1-based start_line and end_line with end_line >= start_line"
		}

		absPath, err := validatePath(args.FilePath)
		if err != nil {
			return fmt.Sprintf("System Error: %v", err)
		}

		content, truncated, err := readFilePortion(absPath, args.StartLine, args.EndLine, getActiveContextLimit())
		if err != nil {
			return fmt.Sprintf("System Error reading path: %v", err)
		}
		if truncated {
			content += "\n[truncated to active context budget; request a smaller line range for more]"
		}
		return content

	case "list_directory":
		var args ListDirArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		if strings.TrimSpace(args.DirPath) == "" {
			return "System Error: dir_path is required"
		}
		absPath, err := validatePath(args.DirPath)
		if err != nil {
			return fmt.Sprintf("System Error: %v", err)
		}

		entries, err := os.ReadDir(absPath)
		if err != nil {
			return fmt.Sprintf("System Error listing directory: %v", err)
		}

		var sb strings.Builder
		for _, entry := range entries {
			if entry.IsDir() {
				fmt.Fprintf(&sb, "%s/\n", entry.Name())
			} else {
				fmt.Fprintf(&sb, "%s\n", entry.Name())
			}
		}
		return sb.String()

	case "grep_search":
		var args GrepSearchArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		if strings.TrimSpace(args.Pattern) == "" {
			return "System Error: pattern is required"
		}
		if args.MaxResults < 0 {
			return "System Error: max_results must not be negative"
		}
		maxResults := args.MaxResults
		if maxResults == 0 {
			maxResults = 50
		}

		re, err := regexp.Compile(args.Pattern)
		if err != nil {
			return fmt.Sprintf("System Error: Invalid regex pattern: %v", err)
		}

		baseDir := workingDir
		if baseDir == "" {
			if cwd, err := os.Getwd(); err == nil {
				baseDir, _ = filepath.Abs(cwd)
			}
		}

		var sb strings.Builder
		resultCount := 0
		truncated := false
		err = filepath.WalkDir(baseDir, func(path string, d os.DirEntry, err error) error {
			if resultCount >= maxResults || truncated {
				return filepath.SkipAll
			}
			if err != nil {
				return err
			}
			if d.IsDir() {
				if d.Name() == ".git" {
					return filepath.SkipDir
				}
				return nil
			}

			info, err := d.Info()
			if err != nil || !info.Mode().IsRegular() || info.Size() > getMaxGrepFileSize() {
				return nil
			}

			if binaryFileExts[strings.ToLower(filepath.Ext(path))] {
				return nil
			}

			content, ok := readGrepFile(path)
			if !ok {
				return nil
			}

			rel, _ := filepath.Rel(baseDir, path)
			for lineNumber, line := range strings.Split(string(content), "\n") {
				if !re.MatchString(line) {
					continue
				}
				result := fmt.Sprintf("%s:%d:%s\n", rel, lineNumber+1, line)
				remaining := 8192 - sb.Len()
				if len(result) > remaining {
					truncated = true
					return filepath.SkipAll
				}
				sb.WriteString(result)
				resultCount++
				if resultCount >= maxResults {
					return filepath.SkipAll
				}
			}
			return nil
		})

		if err != nil {
			return fmt.Sprintf("System Error searching files: %v", err)
		}

		if sb.Len() == 0 {
			return "No matches found."
		}
		if truncated {
			sb.WriteString("[results truncated to preserve context budget; narrow pattern or lower max_results]\n")
		} else if resultCount >= maxResults {
			fmt.Fprintf(&sb, "[results limited to %d matching lines]\n", maxResults)
		}
		return sb.String()
	}

	logDebug("dispatchToolCall failed for: %s", name)
	return fmt.Sprintf("System Error: Tool %s not implemented.", name)
}
