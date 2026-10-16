package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGetRKLLMModelsDirRespectsEnv(t *testing.T) {
	orig := os.Getenv("RKLLM_MODELS_DIR")
	defer os.Setenv("RKLLM_MODELS_DIR", orig)

	customDir := "/tmp/custom-rkllm-models"
	os.Setenv("RKLLM_MODELS_DIR", customDir)

	if got := getRKLLMModelsDir(); got != customDir {
		t.Fatalf("getRKLLMModelsDir() = %q, want %q", got, customDir)
	}
}

func TestGetRKLLMModelsDirDefaultsToUserHome(t *testing.T) {
	orig := os.Getenv("RKLLM_MODELS_DIR")
	defer os.Setenv("RKLLM_MODELS_DIR", orig)

	os.Unsetenv("RKLLM_MODELS_DIR")

	homeDir, err := os.UserHomeDir()
	if err != nil {
		t.Skip("user home dir unavailable")
	}

	expected := filepath.Join(homeDir, "rkllama-models")
	if got := getRKLLMModelsDir(); got != expected {
		t.Fatalf("getRKLLMModelsDir() = %q, want %q", got, expected)
	}
}

func TestGetRKLLMModelPathRespectsEnv(t *testing.T) {
	orig := os.Getenv("RKLLM_MODEL")
	defer os.Setenv("RKLLM_MODEL", orig)

	// Create a temp file to simulate an existing model file
	tmpFile, err := os.CreateTemp("", "test_model_*.rkllm")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()

	os.Setenv("RKLLM_MODEL", tmpFile.Name())

	if got := getRKLLMModelPath(""); got != tmpFile.Name() {
		t.Fatalf("getRKLLMModelPath(\"\") = %q, want %q", got, tmpFile.Name())
	}
}

func TestGetRKLLMModelPathDiscoversSubdirectoryModel(t *testing.T) {
	origModel := os.Getenv("RKLLM_MODEL")
	origDir := os.Getenv("RKLLM_MODELS_DIR")
	defer func() {
		os.Setenv("RKLLM_MODEL", origModel)
		os.Setenv("RKLLM_MODELS_DIR", origDir)
	}()

	os.Unsetenv("RKLLM_MODEL")

	tmpDir, err := os.MkdirTemp("", "models_dir_*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	subDir := filepath.Join(tmpDir, "qwen3-4b-16k")
	if err := os.MkdirAll(subDir, 0755); err != nil {
		t.Fatal(err)
	}

	modelFile := filepath.Join(subDir, "qwen3-4b.rkllm")
	if err := os.WriteFile(modelFile, []byte("dummy"), 0644); err != nil {
		t.Fatal(err)
	}

	os.Setenv("RKLLM_MODELS_DIR", tmpDir)

	if got := getRKLLMModelPath("qwen3-4b"); got != modelFile {
		t.Fatalf("getRKLLMModelPath(\"qwen3-4b\") = %q, want %q", got, modelFile)
	}

	models := listRKLLMModels()
	found := false
	for _, m := range models {
		if strings.Contains(m, "qwen3-4b.rkllm") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("listRKLLMModels() did not list %q: %#v", modelFile, models)
	}
}

func TestCloseRKLLMSessionIdempotent(t *testing.T) {
	// Calling closeRKLLMSession repeatedly should be safe and not panic
	closeRKLLMSession()
	closeRKLLMSession()
}

func TestCaptureStderrCapturesOutput(t *testing.T) {
	msg := "test stderr message 12345"
	captured, err := captureStderr(func() error {
		os.Stderr.WriteString(msg + "\n")
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(captured, msg) {
		t.Fatalf("captureStderr() = %q, want it to contain %q", captured, msg)
	}
}

func TestCaptureStderrRestoresOnPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic to bubble up")
		}
		// Confirm stderr is usable after recover
		if _, err := os.Stderr.WriteString(""); err != nil {
			t.Fatalf("stderr not usable after recovered panic: %v", err)
		}
	}()

	_, _ = captureStderr(func() error {
		os.Stderr.WriteString("about to panic\n")
		panic("boom")
	})
}

func TestAbortRKLLMIdempotent(t *testing.T) {
	// abortRKLLM should be safe and idempotent to call when session is uninitialized or idle
	abortRKLLM()
	abortRKLLM()
}
