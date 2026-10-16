package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFlexibleIntUnmarshal(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		want    int
		wantErr bool
	}{
		{
			name: "numeric literal",
			json: `{"ip":"192.168.1.1","brightness":75}`,
			want: 75,
		},
		{
			name: "string literal",
			json: `{"ip":"192.168.1.1","brightness":"42"}`,
			want: 42,
		},
		{
			name: "string with percent",
			json: `{"ip":"192.168.1.1","brightness":"80%"}`,
			want: 80,
		},
		{
			name:    "invalid string",
			json:    `{"ip":"192.168.1.1","brightness":"high"}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var args SetBrightnessArgs
			err := json.Unmarshal([]byte(tt.json), &args)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if int(args.Brightness) != tt.want {
				t.Fatalf("brightness = %d, want %d", int(args.Brightness), tt.want)
			}
		})
	}
}

func TestValidatePathSecurity(t *testing.T) {
	origWorkingDir := workingDir
	defer func() { workingDir = origWorkingDir }()

	tmpDir, err := os.MkdirTemp("", "validate_path_test_*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	workingDir = tmpDir

	// Allowed inside
	validRel := "file.txt"
	abs, err := validatePath(validRel)
	if err != nil {
		t.Fatalf("expected valid path, got err: %v", err)
	}
	if abs != filepath.Join(tmpDir, "file.txt") {
		t.Fatalf("unexpected abs path: %q", abs)
	}

	// Escaping disallowed
	invalidPaths := []string{
		"../outside.txt",
		"../../etc/passwd",
		"/etc/shadow",
	}
	for _, p := range invalidPaths {
		if _, err := validatePath(p); err == nil {
			t.Fatalf("expected path %q to be rejected, but it was allowed", p)
		}
	}
}

func TestValidatePathRejectsSymlinkedDestination(t *testing.T) {
	origWorkingDir := workingDir
	defer func() { workingDir = origWorkingDir }()

	tmpDir := t.TempDir()
	outsideDir := t.TempDir()
	workingDir = tmpDir

	outsideFile := filepath.Join(outsideDir, "outside.txt")
	if err := os.WriteFile(outsideFile, []byte("secret"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outsideFile, filepath.Join(tmpDir, "linked.txt")); err != nil {
		t.Fatal(err)
	}
	if _, err := validatePath("linked.txt"); err == nil {
		t.Fatal("expected symlinked file destination to be rejected")
	}
}

func TestValidatePathRejectsSymlinkedParentForNewFile(t *testing.T) {
	origWorkingDir := workingDir
	defer func() { workingDir = origWorkingDir }()

	tmpDir := t.TempDir()
	outsideDir := t.TempDir()
	workingDir = tmpDir

	if err := os.Symlink(outsideDir, filepath.Join(tmpDir, "linked-dir")); err != nil {
		t.Fatal(err)
	}
	if _, err := validatePath("linked-dir/new.txt"); err == nil {
		t.Fatal("expected symlinked parent directory to be rejected")
	}
}

func TestReadFilePortionHonorsLineRangeAndBudget(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "lines.txt")
	if err := os.WriteFile(path, []byte("one\ntwo\nthree\nfour\n"), 0600); err != nil {
		t.Fatal(err)
	}

	content, truncated, err := readFilePortion(path, 2, 3, 100)
	if err != nil {
		t.Fatal(err)
	}
	if truncated || content != "two\nthree\n" {
		t.Fatalf("got content %q, truncated=%v", content, truncated)
	}

	content, truncated, err = readFilePortion(path, 0, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !truncated || content != "one" {
		t.Fatalf("got content %q, truncated=%v; want budget-limited output", content, truncated)
	}
}

func TestDispatchToolCallValidatesRequiredArguments(t *testing.T) {
	tests := []struct {
		name string
		args any
		want string
	}{
		{"empty tool name", map[string]any{}, "tool name is required"},
		{"write file path", WriteFileArgs{Content: "x"}, "file_path is required"},
		{"read file path", ReadFileArgs{}, "file_path is required"},
		{"directory path", ListDirArgs{}, "dir_path is required"},
		{"grep pattern", GrepSearchArgs{}, "pattern is required"},
		{"state ip", SetDeviceStateArgs{State: "on"}, "ip is required"},
		{"brightness ip", SetBrightnessArgs{Brightness: 50}, "ip is required"},
		{"sysinfo ip", DeviceIPOnlyArgs{}, "ip is required"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			args, err := json.Marshal(tt.args)
			if err != nil {
				t.Fatal(err)
			}
			toolName := tt.name
			switch tt.name {
			case "empty tool name":
				toolName = ""
			case "write file path":
				toolName = "write_file"
			case "read file path":
				toolName = "read_file"
			case "directory path":
				toolName = "list_directory"
			case "grep pattern":
				toolName = "grep_search"
			case "state ip":
				toolName = "set_device_state"
			case "brightness ip":
				toolName = "set_brightness"
			case "sysinfo ip":
				toolName = "get_sysinfo"
			}
			if got := dispatchToolCall(toolName, args); !strings.Contains(got, tt.want) {
				t.Fatalf("got %q, want error containing %q", got, tt.want)
			}
		})
	}
}

func TestRunShellValidatesAndBoundsOutput(t *testing.T) {
	if got := runShell("   "); got != "System Error: command is required" {
		t.Fatalf("empty command = %q", got)
	}

	got := runShell("printf '%070000s' x")
	if !strings.Contains(got, "[output truncated]") {
		t.Fatalf("expected bounded shell output, got length %d", len(got))
	}
}

func TestReplaceTextRequiresExpectedMatchCount(t *testing.T) {
	tmpDir := t.TempDir()
	origWorkingDir := workingDir
	defer func() { workingDir = origWorkingDir }()
	workingDir = tmpDir
	path := filepath.Join(tmpDir, "replace.txt")
	if err := os.WriteFile(path, []byte("before\nbefore\n"), 0600); err != nil {
		t.Fatal(err)
	}

	args, _ := json.Marshal(ReplaceTextArgs{FilePath: "replace.txt", OldText: "before", NewText: "after"})
	if got := dispatchToolCall("replace_text", args); !strings.Contains(got, "found 2 matches") {
		t.Fatalf("unexpected mismatch result: %q", got)
	}
	content, _ := os.ReadFile(path)
	if string(content) != "before\nbefore\n" {
		t.Fatalf("file changed after mismatch: %q", content)
	}

	args, _ = json.Marshal(ReplaceTextArgs{FilePath: "replace.txt", OldText: "before", NewText: "after", ExpectedMatches: 2})
	if got := dispatchToolCall("replace_text", args); !strings.Contains(got, "replaced 2") {
		t.Fatalf("unexpected success result: %q", got)
	}
	content, _ = os.ReadFile(path)
	if string(content) != "after\nafter\n" {
		t.Fatalf("unexpected replaced content: %q", content)
	}
}

func TestFindFilesReturnsRelativeMatchesAndLimitsResults(t *testing.T) {
	tmpDir := t.TempDir()
	origWorkingDir := workingDir
	defer func() { workingDir = origWorkingDir }()
	workingDir = tmpDir
	if err := os.MkdirAll(filepath.Join(tmpDir, "nested"), 0700); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"a.go", "nested/b.go", "nested/c.txt"} {
		if err := os.WriteFile(filepath.Join(tmpDir, name), []byte("x"), 0600); err != nil {
			t.Fatal(err)
		}
	}

	result, err := findFileMatches(tmpDir, "**/*.go", 10, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result, "a.go\n") || !strings.Contains(result, "nested/b.go\n") {
		t.Fatalf("unexpected matches: %q", result)
	}
	if strings.Contains(result, "c.txt") {
		t.Fatalf("unexpected non-go match: %q", result)
	}
	if !strings.Contains(result, "a.go\n") {
		t.Fatalf("**/ should match files at the root level too: %q", result)
	}

	result, err = findFileMatches(tmpDir, "nested/**/*.go", 10, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result, "nested/b.go\n") || strings.Contains(result, "a.go\n") {
		t.Fatalf("pattern with a directory prefix matched incorrectly: %q", result)
	}
}

func TestRunGitRejectsPathsOutsideWorkingDirectory(t *testing.T) {
	origWorkingDir := workingDir
	defer func() { workingDir = origWorkingDir }()
	workingDir = t.TempDir()
	if got := runGit("diff", "../outside.txt"); !strings.Contains(got, "outside of allowed directory") {
		t.Fatalf("unexpected path validation result: %q", got)
	}
}

func TestApplyPatchValidatesPatchInput(t *testing.T) {
	if got := applyPatch("   "); got != "System Error: patch is required" {
		t.Fatalf("empty patch = %q", got)
	}
	if got := applyPatch(strings.Repeat("x", 256*1024+1)); got != "System Error: patch exceeds the 256 KB limit" {
		t.Fatalf("oversized patch = %q", got)
	}
}

func TestGrepSearchSkipsBinaryAndLargeFiles(t *testing.T) {
	origWorkingDir := workingDir
	defer func() { workingDir = origWorkingDir }()

	tmpDir, err := os.MkdirTemp("", "grep_test_*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	workingDir = tmpDir

	pattern := "TARGET_SECRET_STRING"

	// 1. Regular text file (should match)
	textFile := filepath.Join(tmpDir, "doc.txt")
	if err := os.WriteFile(textFile, []byte("Here is "+pattern+" inside text"), 0644); err != nil {
		t.Fatal(err)
	}

	// 2. Binary extension file (should be skipped)
	binExtFile := filepath.Join(tmpDir, "model.rkllm")
	if err := os.WriteFile(binExtFile, []byte("Here is "+pattern+" inside rkllm"), 0644); err != nil {
		t.Fatal(err)
	}

	// 3. File containing null byte (should be detected as binary and skipped)
	nullByteFile := filepath.Join(tmpDir, "binary_data.dat")
	if err := os.WriteFile(nullByteFile, []byte{0x00, 0x01, 'T', 'A', 'R', 'G', 'E', 'T', '_', 'S', 'E', 'C', 'R', 'E', 'T', '_', 'S', 'T', 'R', 'I', 'N', 'G'}, 0644); err != nil {
		t.Fatal(err)
	}

	rawArgs, _ := json.Marshal(GrepSearchArgs{Pattern: pattern})
	result := dispatchToolCall("grep_search", rawArgs)

	if !strings.Contains(result, "doc.txt:1:Here is "+pattern+" inside text") {
		t.Fatalf("expected doc.txt in grep output, got: %q", result)
	}
	if strings.Contains(result, "model.rkllm") {
		t.Fatalf("expected model.rkllm to be skipped, got: %q", result)
	}
	if strings.Contains(result, "binary_data.dat") {
		t.Fatalf("expected binary_data.dat to be skipped, got: %q", result)
	}
}

func TestGrepSearchLimitsMatchingLines(t *testing.T) {
	tmpDir := t.TempDir()
	origWorkingDir := workingDir
	defer func() { workingDir = origWorkingDir }()
	workingDir = tmpDir

	path := filepath.Join(tmpDir, "matches.txt")
	if err := os.WriteFile(path, []byte("hit one\nhit two\nhit three\n"), 0600); err != nil {
		t.Fatal(err)
	}
	rawArgs, _ := json.Marshal(GrepSearchArgs{Pattern: "hit", MaxResults: 2})
	result := dispatchToolCall("grep_search", rawArgs)
	if !strings.Contains(result, "matches.txt:1:hit one") || !strings.Contains(result, "matches.txt:2:hit two") {
		t.Fatalf("expected first two matching lines, got: %q", result)
	}
	if strings.Contains(result, "matches.txt:3:") {
		t.Fatalf("expected max_results to limit output, got: %q", result)
	}
}

func TestToolsets(t *testing.T) {
	orig := GetActiveToolset()
	defer SetActiveToolset(orig)

	if err := SetActiveToolset("development"); err != nil {
		t.Fatalf("unexpected error setting toolset development: %v", err)
	}
	if GetActiveToolset() != "development" {
		t.Fatalf("expected development toolset, got %q", GetActiveToolset())
	}
	tools := getSystemTools()
	foundOutsource := false
	for _, tool := range tools {
		if tool.Function.Name == "get_devices" {
			t.Fatalf("expected home tool 'get_devices' to be excluded in development toolset")
		}
		if tool.Function.Name == "outsource" {
			foundOutsource = true
		}
	}
	if !foundOutsource {
		t.Fatalf("expected outsource tool in development toolset")
	}

	if err := SetActiveToolset("home"); err != nil {
		t.Fatalf("unexpected error setting toolset home: %v", err)
	}
	if GetActiveToolset() != "home" {
		t.Fatalf("expected home toolset, got %q", GetActiveToolset())
	}
	homeTools := getSystemTools()
	foundGetDevices := false
	for _, tool := range homeTools {
		if tool.Function.Name == "get_devices" {
			foundGetDevices = true
		}
		if tool.Function.Name == "apply_patch" {
			t.Fatalf("expected development tool 'apply_patch' to be excluded in home toolset")
		}
		if tool.Function.Name == "outsource" {
			t.Fatalf("expected development tool 'outsource' to be excluded in home toolset")
		}
	}
	if !foundGetDevices {
		t.Fatalf("expected 'get_devices' tool in home toolset")
	}

	if err := SetActiveToolset("invalid_set"); err == nil {
		t.Fatalf("expected error setting invalid toolset, got nil")
	}

	if err := SetActiveToolset("all"); err != nil {
		t.Fatalf("unexpected error setting toolset all: %v", err)
	}
	if GetActiveToolset() != "all" {
		t.Fatalf("expected all toolset, got %q", GetActiveToolset())
	}
}

func TestOutsourceToolValidation(t *testing.T) {
	origKey := os.Getenv("GOOGLE_API_KEY")
	defer os.Setenv("GOOGLE_API_KEY", origKey)
	os.Unsetenv("GOOGLE_API_KEY")

	// Missing prompt
	rawArgs, _ := json.Marshal(OutsourceArgs{Prompt: ""})
	res := dispatchToolCall("outsource", rawArgs)
	if !strings.Contains(res, "prompt is required") {
		t.Fatalf("expected prompt required error, got: %q", res)
	}

	// Missing API key
	rawArgs, _ = json.Marshal(OutsourceArgs{Prompt: "Hello"})
	res = dispatchToolCall("outsource", rawArgs)
	if !strings.Contains(res, "Google API key is not configured") {
		t.Fatalf("expected API key not configured error, got: %q", res)
	}
}
