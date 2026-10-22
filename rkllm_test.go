package main

import (
	"strings"
	"testing"
)

func TestGetPromptCacheFileDefaultsAndCustom(t *testing.T) {
	// Test default prompt cache file generation
	cacheFile := getPromptCacheFile()
	if cacheFile == "" {
		t.Fatal("expected non-empty prompt cache file path")
	}
	if !strings.HasSuffix(cacheFile, ".rkllm_cache") {
		t.Errorf("expected prompt cache file to end with .rkllm_cache, got %q", cacheFile)
	}
}
