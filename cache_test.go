package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGetCacheFileNamespacedByWorkingDir(t *testing.T) {
	origWorkingDir := workingDir
	origCacheFile := os.Getenv("SCOOTER_CACHE_FILE")
	origCacheDir := os.Getenv("SCOOTER_CACHE_DIR")
	defer func() {
		workingDir = origWorkingDir
		os.Setenv("SCOOTER_CACHE_FILE", origCacheFile)
		os.Setenv("SCOOTER_CACHE_DIR", origCacheDir)
	}()

	os.Unsetenv("SCOOTER_CACHE_FILE")
	os.Unsetenv("SCOOTER_CACHE_DIR")

	workingDir = "/tmp/project_alpha"
	pathA, err := getCacheFile()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	workingDir = "/tmp/project_beta"
	pathB, err := getCacheFile()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if pathA == pathB {
		t.Fatalf("expected distinct cache files for different projects, got both = %q", pathA)
	}

	if !strings.Contains(pathA, "project_alpha") {
		t.Errorf("expected pathA to contain project slug, got %q", pathA)
	}
	if !strings.Contains(pathB, "project_beta") {
		t.Errorf("expected pathB to contain project slug, got %q", pathB)
	}
}

func TestGetCacheFileRespectsEnvOverrides(t *testing.T) {
	origCacheFile := os.Getenv("SCOOTER_CACHE_FILE")
	origCacheDir := os.Getenv("SCOOTER_CACHE_DIR")
	defer func() {
		os.Setenv("SCOOTER_CACHE_FILE", origCacheFile)
		os.Setenv("SCOOTER_CACHE_DIR", origCacheDir)
	}()

	tmpDir, err := os.MkdirTemp("", "cache_test_*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	customFile := filepath.Join(tmpDir, "custom", "my_session.jsonl")
	os.Setenv("SCOOTER_CACHE_FILE", customFile)

	got, err := getCacheFile()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != customFile {
		t.Fatalf("getCacheFile() = %q, want %q", got, customFile)
	}
}
