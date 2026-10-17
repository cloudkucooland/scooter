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
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/cloudkucooland/go-kasa"
	"google.golang.org/genai"
)

var binaryFileExts = map[string]bool{
	".rkllm": true,
	".so":    true,
	".dll":   true,
	".dylib": true,
	".bin":   true,
	".exe":   true,
	".o":     true,
	".a":     true,
	".png":   true,
	".jpg":   true,
	".jpeg":  true,
	".gif":   true,
	".webp":  true,
	".ico":   true,
	".zip":   true,
	".tar":   true,
	".gz":    true,
	".7z":    true,
	".pdf":   true,
	".woff":  true,
	".woff2": true,
	".ttf":   true,
	".eot":   true,
	".mp3":   true,
	".mp4":   true,
	".wav":   true,
}

type Property struct {
	Type        string   `json:"type"`
	Description string   `json:"description"`
	Enum        []string `json:"enum,omitempty"`
}

type Parameters struct {
	Type       string              `json:"type"`
	Properties map[string]Property `json:"properties"`
	Required   []string            `json:"required"`
}

type FunctionDefinition struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Parameters  Parameters `json:"parameters"`
}

type Tool struct {
	Type     string             `json:"type"`
	Function FunctionDefinition `json:"function"`
}

var activeToolset = "all"

func GetActiveToolset() string {
	if activeToolset == "" {
		return "all"
	}
	return activeToolset
}

func SetActiveToolset(name string) error {
	normalized := strings.ToLower(strings.TrimSpace(name))
	switch normalized {
	case "dev", "development":
		activeToolset = "development"
	case "home", "kasa", "iot":
		activeToolset = "home"
	case "web", "net", "internet":
		activeToolset = "web"
	case "all", "every":
		activeToolset = "all"
	default:
		return fmt.Errorf("unknown toolset %q (available: development, home, web, all)", name)
	}
	return nil
}

func listToolsetsNames() []string {
	return []string{
		"development (dev)",
		"home (kasa, iot)",
		"web (internet)",
		"all (default)",
	}
}

func getWebTools() []Tool {
	return []Tool{
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "web_fetch",
				Description: "Processes content from URL(s) embedded in a prompt. Include up to 20 URLs and instructions (e.g., summarize, extract specific data) directly in the 'prompt' parameter.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"prompt": {
							Type:        "string",
							Description: "A prompt containing URL(s) (up to 20) to fetch and instructions on how to process their content. URLs must start with http:// or https://.",
						},
					},
					Required: []string{"prompt"},
				},
			},
		},
	}
}

func getDevelopmentTools() []Tool {
	return []Tool{
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "apply_patch",
				Description: "Applies a structured patch to files within the working directory.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"patch": {Type: "string", Description: "Patch text using Begin Patch/End Patch markers."},
					},
					Required: []string{"patch"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "git_status",
				Description: "Shows the Git working tree status for the working directory.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"path": {Type: "string", Description: "Optional path inside the working directory to limit the status query."},
					},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "git_diff",
				Description: "Shows the Git diff for the working directory or one file.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"path": {Type: "string", Description: "Optional path inside the working directory to limit the diff."},
					},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "find_files",
				Description: "Finds files recursively under the working directory using a glob pattern.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"pattern":     {Type: "string", Description: "Glob pattern such as '*.go' or '**/*_test.go'."},
						"max_results": {Type: "integer", Description: "Optional maximum number of paths to return; defaults to 100."},
					},
					Required: []string{"pattern"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "replace_text",
				Description: "Replaces exact text in a file within the working directory. Fails unless the expected match count is met.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"file_path":        {Type: "string", Description: "Path of the file to edit."},
						"old_text":         {Type: "string", Description: "Exact text to replace."},
						"new_text":         {Type: "string", Description: "Replacement text."},
						"expected_matches": {Type: "integer", Description: "Expected number of exact matches; defaults to 1."},
					},
					Required: []string{"file_path", "old_text", "new_text"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "shell",
				Description: "Runs a shell command as the current user in the working directory. Use only when a local shell command is necessary.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"command": {Type: "string", Description: "Shell command to execute."},
					},
					Required: []string{"command"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "write_file",
				Description: "Writes text contents directly to a file on disk instead of outputting chat text.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"file_path": {Type: "string", Description: "Path of the file to write (relative to working directory or absolute within it)."},
						"content":   {Type: "string", Description: "Raw text or code contents for the file."},
					},
					Required: []string{"file_path", "content"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "read_file",
				Description: "Reads the content of a file on disk.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"file_path":  {Type: "string", Description: "Path of the file to read (relative to working directory or absolute within it)."},
						"start_line": {Type: "integer", Description: "Optional 1-based first line to read."},
						"end_line":   {Type: "integer", Description: "Optional inclusive last line to read."},
					},
					Required: []string{"file_path"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "list_directory",
				Description: "Lists files and subdirectories in a given directory.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"dir_path": {Type: "string", Description: "Directory path to list (relative to working directory or absolute within it; '.' for current directory)."},
					},
					Required: []string{"dir_path"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "outsource",
				Description: "Outsources a complex task or prompt to Google's Gemini GenAI API when the local model needs advanced cloud capabilities.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"prompt": {Type: "string", Description: "The prompt or instructions to send to Google GenAI."},
					},
					Required: []string{"prompt"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "grep_search",
				Description: "Recursively searches for a regex pattern in files within the current directory.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"pattern":     {Type: "string", Description: "The regex pattern to search for."},
						"max_results": {Type: "integer", Description: "Optional maximum number of matching lines to return; defaults to 50."},
					},
					Required: []string{"pattern"},
				},
			},
		},
	}
}

func getHomeTools() []Tool {
	return []Tool{
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "get_devices",
				Description: "Reports all smart-home device status.",
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "set_device_state",
				Description: "Sets the state (on/off) of a smart-home device.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"ip":    {Type: "string", Description: "IP address of the device."},
						"state": {Type: "string", Description: "Desired state: 'on' or 'off'."},
					},
					Required: []string{"ip", "state"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "set_brightness",
				Description: "Sets the brightness of a dimmable smart light or switch (1-100).",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"ip":         {Type: "string", Description: "IP address of the device."},
						"brightness": {Type: "integer", Description: "Brightness level from 1 to 100."},
					},
					Required: []string{"ip", "brightness"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "get_sysinfo",
				Description: "Retrieves detailed system parameters and status from a specific device.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"ip": {Type: "string", Description: "IP address of the device."},
					},
					Required: []string{"ip"},
				},
			},
		},
	}
}

func getSystemTools() []Tool {
	switch GetActiveToolset() {
	case "development":
		return getDevelopmentTools()
	case "home":
		return getHomeTools()
	case "web":
		return getWebTools()
	case "all":
		fallthrough
	default:
		return append(append(getDevelopmentTools(), getHomeTools()...), getWebTools()...)
	}
}

type WebFetchArgs struct {
	Prompt string `json:"prompt"`
}

func webFetch(prompt string) string {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return "System Error: prompt is required"
	}

	urlRegex := regexp.MustCompile(`https?://[^\s"'<>]+`)
	matches := urlRegex.FindAllString(prompt, 20)
	if len(matches) == 0 {
		if strings.HasPrefix(prompt, "http://") || strings.HasPrefix(prompt, "https://") {
			matches = []string{prompt}
		} else {
			return "System Error: no valid HTTP/HTTPS URLs found in prompt"
		}
	}

	seen := make(map[string]bool)
	var uniqueUrls []string
	for _, u := range matches {
		u = strings.TrimRight(u, ".,;:!?)")
		if !seen[u] {
			seen[u] = true
			uniqueUrls = append(uniqueUrls, u)
		}
	}

	var sb strings.Builder
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	for _, u := range uniqueUrls {
		req, err := http.NewRequestWithContext(context.Background(), "GET", u, nil)
		if err != nil {
			fmt.Fprintf(&sb, "<source url=\"%s\">\nError creating request: %v\n</source>\n", sanitizeXmlForSource(u), err)
			continue
		}
		req.Header.Set("User-Agent", "Scooter/1.0 (+https://github.com/cloudkucooland/scooter)")

		resp, err := client.Do(req)
		if err != nil {
			fmt.Fprintf(&sb, "<source url=\"%s\">\nError fetching URL: %v\n</source>\n", sanitizeXmlForSource(u), err)
			continue
		}
		defer resp.Body.Close()

		bodyBytes, err := io.ReadAll(io.LimitReader(resp.Body, 1024*1024))
		if err != nil {
			fmt.Fprintf(&sb, "<source url=\"%s\">\nError reading response body: %v\n</source>\n", sanitizeXmlForSource(u), err)
			continue
		}

		content := string(bodyBytes)
		if len(content) > 100*1024 {
			content = content[:100*1024] + "\n[content truncated at 100KB]"
		}

		fmt.Fprintf(&sb, "<source url=\"%s\">\n%s\n</source>\n", sanitizeXmlForSource(u), sanitizeXmlForSource(content))
	}

	return sb.String()
}

func sanitizeXmlForSource(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	return s
}

// FlexibleInt supports unmarshaling integers from JSON numeric literals or quoted strings.
type FlexibleInt int

func (fi *FlexibleInt) UnmarshalJSON(data []byte) error {
	data = bytes.TrimSpace(data)
	if len(data) == 0 || bytes.Equal(data, []byte("null")) {
		*fi = 0
		return nil
	}
	if data[0] == '"' {
		var s string
		if err := json.Unmarshal(data, &s); err != nil {
			return err
		}
		s = strings.TrimSpace(s)
		s = strings.TrimSuffix(s, "%")
		var val int
		if _, err := fmt.Sscanf(s, "%d", &val); err != nil {
			return fmt.Errorf("invalid integer: %q", s)
		}
		*fi = FlexibleInt(val)
		return nil
	}
	var val int
	if err := json.Unmarshal(data, &val); err != nil {
		return err
	}
	*fi = FlexibleInt(val)
	return nil
}

type WriteFileArgs struct {
	FilePath string `json:"file_path"`
	Content  string `json:"content"`
}

type ShellArgs struct {
	Command string `json:"command"`
}

type ReplaceTextArgs struct {
	FilePath        string `json:"file_path"`
	OldText         string `json:"old_text"`
	NewText         string `json:"new_text"`
	ExpectedMatches int    `json:"expected_matches,omitempty"`
}

type FindFilesArgs struct {
	Pattern    string `json:"pattern"`
	MaxResults int    `json:"max_results,omitempty"`
}

type GitPathArgs struct {
	Path string `json:"path,omitempty"`
}

type ApplyPatchArgs struct {
	Patch string `json:"patch"`
}

type ReadFileArgs struct {
	FilePath  string `json:"file_path"`
	StartLine int    `json:"start_line,omitempty"`
	EndLine   int    `json:"end_line,omitempty"`
}

type ListDirArgs struct {
	DirPath string `json:"dir_path"`
}

type GrepSearchArgs struct {
	Pattern    string `json:"pattern"`
	MaxResults int    `json:"max_results,omitempty"`
}

type OutsourceArgs struct {
	Prompt string `json:"prompt"`
}

type SetDeviceStateArgs struct {
	IP    string `json:"ip"`
	State string `json:"state"`
}

type SetBrightnessArgs struct {
	IP         string      `json:"ip"`
	Brightness FlexibleInt `json:"brightness"`
}

type DeviceIPOnlyArgs struct {
	IP string `json:"ip"`
}

func validatePath(path string) (string, error) {
	baseDir := workingDir
	if baseDir == "" {
		if cwd, err := os.Getwd(); err == nil {
			baseDir, _ = filepath.Abs(cwd)
		}
	}

	resolvedBase, err := filepath.EvalSymlinks(baseDir)
	if err != nil {
		return "", fmt.Errorf("resolve allowed directory: %w", err)
	}

	var targetPath string
	if filepath.IsAbs(path) {
		targetPath = filepath.Clean(path)
	} else {
		targetPath = filepath.Join(baseDir, path)
	}

	// Resolve the final destination when it exists. For a new file, resolve
	// the nearest existing parent so a symlinked directory cannot escape the
	// working directory before the file is created.
	resolvedTarget, err := resolveExistingPath(targetPath)
	if err != nil {
		return "", err
	}

	rel, err := filepath.Rel(resolvedBase, resolvedTarget)
	if err != nil {
		return "", err
	}

	if strings.HasPrefix(rel, "..") || rel == ".." {
		return "", fmt.Errorf("path %s is outside of allowed directory %s", targetPath, resolvedBase)
	}

	return targetPath, nil
}

func resolveExistingPath(path string) (string, error) {
	path = filepath.Clean(path)
	current := path
	var suffix []string
	for {
		resolved, err := filepath.EvalSymlinks(current)
		if err == nil {
			for i := len(suffix) - 1; i >= 0; i-- {
				resolved = filepath.Join(resolved, suffix[i])
			}
			return resolved, nil
		}
		if !os.IsNotExist(err) {
			return "", fmt.Errorf("resolve path %s: %w", path, err)
		}
		parent := filepath.Dir(current)
		if parent == current {
			return "", fmt.Errorf("resolve path %s: %w", path, err)
		}
		suffix = append(suffix, filepath.Base(current))
		current = parent
	}
}

func readFilePortion(path string, startLine, endLine, tokenBudget int) (string, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", false, err
	}
	defer f.Close()

	maxBytes := tokenBudget * 3
	if maxBytes < 1 {
		return "", true, nil
	}

	scanner := bufio.NewScanner(f)
	// Permit reasonably large source lines while still bounding the returned
	// content by maxBytes below.
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	var out strings.Builder
	lineNumber := 0
	truncated := false
	for scanner.Scan() {
		lineNumber++
		if startLine > 0 && lineNumber < startLine {
			continue
		}
		if endLine > 0 && lineNumber > endLine {
			break
		}

		line := scanner.Text() + "\n"
		remaining := maxBytes - out.Len()
		if len(line) > remaining {
			out.WriteString(line[:remaining])
			truncated = true
			break
		}
		out.WriteString(line)
	}
	if err := scanner.Err(); err != nil {
		return "", false, err
	}
	return out.String(), truncated, nil
}

// readGrepFile reads one candidate file and closes it before returning. The
// helper keeps file descriptors bounded while filepath.WalkDir visits a large
// tree.
func readGrepFile(path string) ([]byte, bool) {
	f, err := os.Open(path)
	if err != nil {
		return nil, false
	}

	header := make([]byte, 512)
	n, headerErr := f.Read(header)
	if headerErr != nil && headerErr != io.EOF {
		_ = f.Close()
		return nil, false
	}
	if bytes.IndexByte(header[:n], 0) != -1 {
		_ = f.Close()
		return nil, false
	}

	if _, err := f.Seek(0, io.SeekStart); err != nil {
		_ = f.Close()
		return nil, false
	}
	content, readErr := io.ReadAll(io.LimitReader(f, getMaxGrepFileSize()))
	closeErr := f.Close()
	if readErr != nil || closeErr != nil {
		return nil, false
	}
	return content, true
}

type cappedBuffer struct {
	data      []byte
	max       int
	truncated bool
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	remaining := b.max - len(b.data)
	if remaining <= 0 {
		b.truncated = true
		return len(p), nil
	}
	if len(p) > remaining {
		b.data = append(b.data, p[:remaining]...)
		b.truncated = true
		return len(p), nil
	}
	b.data = append(b.data, p...)
	return len(p), nil
}

func runShell(command string) string {
	command = strings.TrimSpace(command)
	if command == "" {
		return "System Error: command is required"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	cmd.Dir = workingDir
	output := &cappedBuffer{max: getMaxShellOutputSize()}
	cmd.Stdout = output
	cmd.Stderr = output
	err := cmd.Run()
	result := string(output.data)
	if output.truncated {
		result += "\n[output truncated]"
	}
	if ctx.Err() == context.DeadlineExceeded {
		return result + "\nSystem Error: command timed out after 30 seconds"
	}
	if err != nil {
		return result + fmt.Sprintf("\nSystem Error: command failed: %v", err)
	}
	return result
}

func findFileMatches(baseDir, pattern string, maxResults, maxBytes int) (string, error) {
	var out strings.Builder
	count := 0
	truncated := false
	err := filepath.WalkDir(baseDir, func(filePath string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			if entry.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		rel, err := filepath.Rel(baseDir, filePath)
		if err != nil {
			return nil
		}
		rel = filepath.ToSlash(rel)
		matched, err := globMatch(pattern, rel)
		if !matched && !strings.Contains(pattern, "/") {
			matched, err = globMatch(pattern, filepath.Base(rel))
		}
		if err != nil {
			return err
		}
		if !matched {
			return nil
		}

		line := rel + "\n"
		if out.Len()+len(line) > maxBytes {
			truncated = true
			return filepath.SkipAll
		}
		out.WriteString(line)
		count++
		if count >= maxResults {
			return filepath.SkipAll
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	if truncated {
		out.WriteString("[results truncated to the active context budget]\n")
	} else if count >= maxResults {
		fmt.Fprintf(&out, "[results limited to %d files]\n", maxResults)
	}
	return out.String(), nil
}

// globMatch matches slash-separated paths. A segment equal to ** matches zero
// or more complete path segments; other segments use path.Match semantics.
func globMatch(pattern, name string) (bool, error) {
	patternParts := strings.Split(strings.Trim(pattern, "/"), "/")
	nameParts := strings.Split(strings.Trim(name, "/"), "/")
	type position struct{ pattern, name int }
	known := make(map[position]bool)
	results := make(map[position]bool)
	var match func(int, int) (bool, error)
	match = func(pi, ni int) (bool, error) {
		pos := position{pi, ni}
		if known[pos] {
			return results[pos], nil
		}
		known[pos] = true
		if pi == len(patternParts) {
			results[pos] = ni == len(nameParts)
			return results[pos], nil
		}

		if patternParts[pi] == "**" {
			ok, err := match(pi+1, ni) // ** consumes no path segment
			if err != nil || ok {
				results[pos] = ok
				return ok, err
			}
			if ni < len(nameParts) {
				ok, err = match(pi, ni+1) // ** consumes one segment
				results[pos] = ok
				return ok, err
			}
			return false, nil
		}

		if ni >= len(nameParts) {
			return false, nil
		}
		segmentMatches, err := path.Match(patternParts[pi], nameParts[ni])
		if err != nil || !segmentMatches {
			return false, err
		}
		ok, err := match(pi+1, ni+1)
		results[pos] = ok
		return ok, err
	}
	return match(0, 0)
}

func runGit(subcommand string, pathArg string) string {
	args := []string{subcommand}
	if strings.TrimSpace(pathArg) != "" {
		absPath, err := validatePath(pathArg)
		if err != nil {
			return fmt.Sprintf("System Error: %v", err)
		}
		relPath, err := filepath.Rel(workingDir, absPath)
		if err != nil {
			return fmt.Sprintf("System Error resolving Git path: %v", err)
		}
		args = append(args, "--", relPath)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = workingDir
	output := &cappedBuffer{max: getMaxShellOutputSize()}
	cmd.Stdout = output
	cmd.Stderr = output
	err := cmd.Run()
	result := string(output.data)
	if output.truncated {
		result += "\n[output truncated]"
	}
	if ctx.Err() == context.DeadlineExceeded {
		return result + "\nSystem Error: git command timed out after 30 seconds"
	}
	if err != nil {
		return result + fmt.Sprintf("\nSystem Error: git command failed: %v", err)
	}
	if result == "" {
		return "No Git output."
	}
	return result
}

func applyPatch(patchText string) string {
	if strings.TrimSpace(patchText) == "" {
		return "System Error: patch is required"
	}
	if len(patchText) > 256*1024 {
		return "System Error: patch exceeds the 256 KB limit"
	}
	patchCommand, err := exec.LookPath("apply_patch")
	if err != nil {
		return fmt.Sprintf("System Error: apply_patch executable not found: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, patchCommand)
	cmd.Dir = workingDir
	cmd.Stdin = strings.NewReader(patchText)
	output := &cappedBuffer{max: getMaxShellOutputSize()}
	cmd.Stdout = output
	cmd.Stderr = output
	err = cmd.Run()
	result := string(output.data)
	if output.truncated {
		result += "\n[output truncated]"
	}
	if ctx.Err() == context.DeadlineExceeded {
		return result + "\nSystem Error: patch timed out after 30 seconds"
	}
	if err != nil {
		return result + fmt.Sprintf("\nSystem Error: patch failed: %v", err)
	}
	if result == "" {
		return "Patch applied successfully."
	}
	return result
}

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
		return webFetch(args.Prompt)

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
		result, err := findFileMatches(baseDir, filepath.ToSlash(args.Pattern), maxResults, getActiveContextLimit()*3)
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

		// Ensure target directory tree exists prior to writing
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
			content += "\n[truncated to the active context budget; request a smaller line range for more]"
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
				// Skip .git directory and hidden dirs
				if d.Name() == ".git" {
					return filepath.SkipDir
				}
				return nil
			}

			// Skip non-regular files and files exceeding 2 MB limit
			info, err := d.Info()
			if err != nil || !info.Mode().IsRegular() || info.Size() > getMaxGrepFileSize() {
				return nil
			}

			// Skip known binary extensions
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
				remaining := getActiveContextLimit()*3 - sb.Len()
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
			sb.WriteString("[results truncated to the active context budget; narrow the pattern or lower max_results]\n")
		} else if resultCount >= maxResults {
			fmt.Fprintf(&sb, "[results limited to %d matching lines]\n", maxResults)
		}
		return sb.String()

	case "get_devices":
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		d, err := kasa.BroadcastDiscovery(ctx, 2)
		if err != nil {
			return fmt.Sprintf("System Error doing kasa discovery: %v", err)
		}

		var sb strings.Builder
		fmt.Fprintf(&sb, "Device\tIP/ID:\tState\tBrightness\n")

		for k, v := range d {
			if len(v.Children) == 0 {
				fmt.Fprintf(&sb, "%s\t%s\t%d\t\t%3d\n", v.Alias, k, v.RelayState, v.Brightness)
			}
			// restore child logic, but exceeds the context size
		}

		if sb.Len() == 0 {
			return "No Kasa smart home devices discovered on the local network."
		}

		return sb.String()

	case "set_device_state":
		var args SetDeviceStateArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		if strings.TrimSpace(args.IP) == "" {
			return "System Error: ip is required"
		}

		targetState := strings.ToLower(strings.TrimSpace(args.State))
		var turnOn bool
		if targetState == "on" || targetState == "1" || targetState == "true" {
			turnOn = true
		} else if targetState == "off" || targetState == "0" || targetState == "false" {
			turnOn = false
		} else {
			return fmt.Sprintf("System Error: Invalid state '%s', use 'on' or 'off'.", args.State)
		}

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		dev, err := kasa.NewDevice(args.IP)
		if err != nil || dev == nil {
			return fmt.Sprintf("System Error creating Kasa device instance for IP %s: %v", args.IP, err)
		}

		if err := dev.SetRelayStateCtx(ctx, turnOn); err != nil {
			return fmt.Sprintf("System Error setting device state: %v", err)
		}

		return fmt.Sprintf("Success: Set device %s to %s.", args.IP, targetState)

	case "set_brightness":
		var args SetBrightnessArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		if strings.TrimSpace(args.IP) == "" {
			return "System Error: ip is required"
		}

		brightness := int(args.Brightness)
		if brightness < 1 || brightness > 100 {
			return fmt.Sprintf("System Error: Brightness must be a number between 1 and 100, got %d", brightness)
		}

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		dev, err := kasa.NewDevice(args.IP)
		if err != nil || dev == nil {
			return fmt.Sprintf("System Error creating Kasa instance for IP %s: %v", args.IP, err)
		}

		if err := dev.SetBrightnessCtx(ctx, brightness); err != nil {
			return fmt.Sprintf("System Error setting brightness: %v", err)
		}

		return fmt.Sprintf("Success: Set brightness for %s to %d%%.", args.IP, brightness)

	case "get_sysinfo":
		var args DeviceIPOnlyArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		if strings.TrimSpace(args.IP) == "" {
			return "System Error: ip is required"
		}

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		dev, err := kasa.NewDevice(args.IP)
		if err != nil || dev == nil {
			return fmt.Sprintf("System Error creating Kasa instance for IP %s: %v", args.IP, err)
		}

		sysInfo, err := dev.GetSettingsCtx(ctx)
		if err != nil {
			return fmt.Sprintf("System Error fetching sysinfo: %v", err)
		}

		infoJSON, _ := json.MarshalIndent(sysInfo, "", "  ")
		return string(infoJSON)
	}

	logDebug("dispatchToolCall failed for: %s", name)
	return fmt.Sprintf("System Error: Tool %s not implemented.", name)
}
