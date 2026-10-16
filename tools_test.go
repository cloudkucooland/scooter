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

	if !strings.Contains(result, "doc.txt") {
		t.Fatalf("expected doc.txt in grep output, got: %q", result)
	}
	if strings.Contains(result, "model.rkllm") {
		t.Fatalf("expected model.rkllm to be skipped, got: %q", result)
	}
	if strings.Contains(result, "binary_data.dat") {
		t.Fatalf("expected binary_data.dat to be skipped, got: %q", result)
	}
}
