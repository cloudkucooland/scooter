package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

func webFetch(prompt string, rawArgs json.RawMessage) string {
	prompt = strings.TrimSpace(prompt)

	// Fallback to "url" key if "prompt" key was empty in JSON
	if prompt == "" && len(rawArgs) > 0 {
		var generic map[string]interface{}
		if err := json.Unmarshal(rawArgs, &generic); err == nil {
			if u, ok := generic["url"].(string); ok {
				prompt = u
			}
		}
	}

	if prompt == "" {
		return "System Error: prompt or url parameter is required"
	}

	urlRegex := regexp.MustCompile(`https?://[^\s"'<>]+`)
	matches := urlRegex.FindAllString(prompt, 20)

	// Fallback: If no explicit http(s):// found, check if a domain was passed (e.g. "indievisible.org")
	if len(matches) == 0 {
		domainRegex := regexp.MustCompile(`\b([a-zA-Z0-9-]+\.[a-zA-Z]{2,})\b`)
		if domainMatches := domainRegex.FindAllString(prompt, 1); len(domainMatches) > 0 {
			matches = []string{"https://" + domainMatches[0]}
		} else if strings.HasPrefix(prompt, "http://") || strings.HasPrefix(prompt, "https://") {
			matches = []string{prompt}
		} else {
			// As a last resort, assume the whole prompt string is a URL or domain
			matches = []string{"https://" + strings.TrimPrefix(prompt, "http://")}
		}
	}

	seen := make(map[string]bool)
	var uniqueUrls []string
	for _, u := range matches {
		u = strings.TrimRight(u, ".,;:!?)")
		if !strings.HasPrefix(u, "http://") && !strings.HasPrefix(u, "https://") {
			u = "https://" + u
		}
		if !seen[u] {
			seen[u] = true
			uniqueUrls = append(uniqueUrls, u)
		}
	}

	var sb strings.Builder
	dialer := &net.Dialer{Timeout: 10 * time.Second}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.DialContext = func(ctx context.Context, network string, address string) (net.Conn, error) {
		host, port, err := net.SplitHostPort(address)
		if err != nil {
			return nil, err
		}
		ips, err := net.DefaultResolver.LookupIPAddr(ctx, host)
		if err != nil {
			return nil, err
		}
		if len(ips) == 0 {
			return nil, fmt.Errorf("host %q resolved to no addresses", host)
		}
		for _, resolved := range ips {
			if prohibitedFetchIP(resolved.IP) {
				return nil, fmt.Errorf("host %q resolves to prohibited address %s", host, resolved.IP)
			}
		}
		var lastErr error
		for _, resolved := range ips {
			conn, dialErr := dialer.DialContext(ctx, network, net.JoinHostPort(resolved.IP.String(), port))
			if dialErr == nil {
				return conn, nil
			}
			lastErr = dialErr
		}
		return nil, lastErr
	}
	client := &http.Client{
		Timeout:   30 * time.Second,
		Transport: transport,
	}

	for _, u := range uniqueUrls {
		parsed, err := url.Parse(u)
		if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Hostname() == "" || parsed.User != nil {
			fmt.Fprintf(&sb, "<source url=\"%s\">\nError fetching URL: invalid HTTP/HTTPS URL\n</source>\n", sanitizeXmlForSource(u))
			continue
		}
		if ip := net.ParseIP(parsed.Hostname()); ip != nil && prohibitedFetchIP(ip) {
			fmt.Fprintf(&sb, "<source url=\"%s\">\nError fetching URL: prohibited address\n</source>\n", sanitizeXmlForSource(u))
			continue
		}
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
		bodyBytes, err := io.ReadAll(io.LimitReader(resp.Body, 1024*1024))
		_ = resp.Body.Close()
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

func prohibitedFetchIP(ip net.IP) bool {
	if ip == nil || ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsUnspecified() || ip.IsMulticast() || !ip.IsGlobalUnicast() {
		return true
	}
	for _, cidr := range []string{
		"0.0.0.0/8", "100.64.0.0/10", "192.0.0.0/24", "192.0.2.0/24",
		"198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24", "240.0.0.0/4",
		"2001:db8::/32",
	} {
		_, network, _ := net.ParseCIDR(cidr)
		if network.Contains(ip) {
			return true
		}
	}
	return false
}

func sanitizeXmlForSource(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	return s
}

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
	if maxBytes > 16384 {
		maxBytes = 16384
	}
	if maxBytes < 1 {
		return "", true, nil
	}

	scanner := bufio.NewScanner(f)
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
		out.WriteString("[results truncated to active context budget]\n")
	} else if count >= maxResults {
		fmt.Fprintf(&out, "[results limited to %d files]\n", maxResults)
	}
	return out.String(), nil
}

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
			ok, err := match(pi+1, ni)
			if err != nil || ok {
				results[pos] = ok
				return ok, err
			}
			if ni < len(nameParts) {
				ok, err = match(pi, ni+1)
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

func fileMetadata(absPath string) string {
	info, err := os.Lstat(absPath)
	if err != nil {
		return fmt.Sprintf("System Error: %v", err)
	}

	isSymlink := info.Mode()&os.ModeSymlink != 0
	symlinkTarget := ""
	if isSymlink {
		if target, err := os.Readlink(absPath); err == nil {
			symlinkTarget = target
		}
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "Path: %s\n", absPath)
	fmt.Fprintf(&sb, "Size: %d bytes\n", info.Size())
	fmt.Fprintf(&sb, "Mode: %s\n", info.Mode().String())
	fmt.Fprintf(&sb, "Permissions: %#o\n", info.Mode().Perm())
	fmt.Fprintf(&sb, "ModTime: %s\n", info.ModTime().Format(time.RFC3339))
	fmt.Fprintf(&sb, "IsDir: %v\n", info.IsDir())
	fmt.Fprintf(&sb, "IsSymlink: %v\n", isSymlink)
	if symlinkTarget != "" {
		fmt.Fprintf(&sb, "SymlinkTarget: %s\n", symlinkTarget)
	}
	return sb.String()
}

func generateDiff(original, modified string) string {
	origLines := strings.Split(original, "\n")
	modLines := strings.Split(modified, "\n")
	var sb strings.Builder
	max := len(origLines)
	if len(modLines) > max {
		max = len(modLines)
	}
	for i := 0; i < max; i++ {
		var origLine, modLine string
		hasOrig := i < len(origLines)
		hasMod := i < len(modLines)
		if hasOrig {
			origLine = origLines[i]
		}
		if hasMod {
			modLine = modLines[i]
		}
		if hasOrig && hasMod && origLine != modLine {
			fmt.Fprintf(&sb, "- %s\n+ %s\n", origLine, modLine)
		} else if hasOrig && !hasMod {
			fmt.Fprintf(&sb, "- %s\n", origLine)
		} else if !hasOrig && hasMod {
			fmt.Fprintf(&sb, "+ %s\n", modLine)
		}
	}
	return sb.String()
}

func editFile(absPath string, operations []EditOperation, previewOnly bool) (string, error) {
	contentBytes, err := os.ReadFile(absPath)
	if err != nil {
		return "", fmt.Errorf("read file: %w", err)
	}
	originalContent := string(contentBytes)
	modifiedContent := originalContent

	var replaceOps []EditOperation
	var insertOps []EditOperation
	var deleteOps []EditOperation

	for _, op := range operations {
		switch op.Type {
		case "replace":
			replaceOps = append(replaceOps, op)
		case "insert":
			insertOps = append(insertOps, op)
		case "delete":
			deleteOps = append(deleteOps, op)
		default:
			return "", fmt.Errorf("unknown operation type: %q", op.Type)
		}
	}

	for _, op := range replaceOps {
		if op.Pattern == "" {
			return "", fmt.Errorf("replace operation missing pattern")
		}
		re, err := regexp.Compile(op.Pattern)
		if err != nil {
			re, err = regexp.Compile(regexp.QuoteMeta(op.Pattern))
			if err != nil {
				return "", fmt.Errorf("invalid pattern %q: %w", op.Pattern, err)
			}
		}
		modifiedContent = re.ReplaceAllString(modifiedContent, op.Replacement)
	}

	if len(insertOps) > 0 || len(deleteOps) > 0 {
		lines := strings.Split(modifiedContent, "\n")

		sortedDeletions := make([]EditOperation, len(deleteOps))
		copy(sortedDeletions, deleteOps)
		for i := 0; i < len(sortedDeletions); i++ {
			for j := i + 1; j < len(sortedDeletions); j++ {
				if sortedDeletions[i].StartLine < sortedDeletions[j].StartLine {
					sortedDeletions[i], sortedDeletions[j] = sortedDeletions[j], sortedDeletions[i]
				}
			}
		}

		for _, op := range sortedDeletions {
			start := op.StartLine - 1
			end := op.EndLine - 1
			if end < start {
				end = start
			}
			if start < 0 {
				start = 0
			}
			if start >= len(lines) {
				continue
			}
			if end >= len(lines) {
				end = len(lines) - 1
			}
			lines = append(lines[:start], lines[end+1:]...)
		}

		sortedInsertions := make([]EditOperation, len(insertOps))
		copy(sortedInsertions, insertOps)
		for i := 0; i < len(sortedInsertions); i++ {
			for j := i + 1; j < len(sortedInsertions); j++ {
				if sortedInsertions[i].LineNumber < sortedInsertions[j].LineNumber {
					sortedInsertions[i], sortedInsertions[j] = sortedInsertions[j], sortedInsertions[i]
				}
			}
		}

		for _, op := range sortedInsertions {
			var insertLines []string
			switch c := op.Content.(type) {
			case string:
				insertLines = strings.Split(strings.TrimSuffix(c, "\n"), "\n")
			case []interface{}:
				for _, item := range c {
					if s, ok := item.(string); ok {
						insertLines = append(insertLines, strings.TrimSuffix(s, "\n"))
					}
				}
			case []string:
				for _, s := range c {
					insertLines = append(insertLines, strings.TrimSuffix(s, "\n"))
				}
			}

			idx := op.LineNumber
			if idx == -1 {
				idx = len(lines) + 1
			}
			if idx <= 0 {
				idx = 1
			}
			targetIdx := idx - 1
			if targetIdx > len(lines) {
				targetIdx = len(lines)
			}

			newLines := make([]string, 0, len(lines)+len(insertLines))
			newLines = append(newLines, lines[:targetIdx]...)
			newLines = append(newLines, insertLines...)
			newLines = append(newLines, lines[targetIdx:]...)
			lines = newLines
		}

		modifiedContent = strings.Join(lines, "\n")
	}

	if originalContent == modifiedContent {
		return "No changes made (content unchanged).", nil
	}

	diff := generateDiff(originalContent, modifiedContent)

	if previewOnly {
		return fmt.Sprintf("Preview Diff:\n%s", diff), nil
	}

	dir := filepath.Dir(absPath)
	tmpFile, err := os.CreateTemp(dir, "scooter-edit-*")
	if err != nil {
		return "", fmt.Errorf("create temp file: %w", err)
	}
	tmpName := tmpFile.Name()
	defer func() {
		_ = tmpFile.Close()
		_ = os.Remove(tmpName)
	}()

	if _, err := tmpFile.WriteString(modifiedContent); err != nil {
		return "", fmt.Errorf("write temp file: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		return "", fmt.Errorf("close temp file: %w", err)
	}

	info, err := os.Stat(absPath)
	if err == nil {
		_ = os.Chmod(tmpName, info.Mode())
	}

	if err := os.Rename(tmpName, absPath); err != nil {
		return "", fmt.Errorf("atomic rename: %w", err)
	}

	return fmt.Sprintf("Success: File edited successfully.\nDiff:\n%s", diff), nil
}
