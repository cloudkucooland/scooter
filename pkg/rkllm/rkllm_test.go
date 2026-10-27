package rkllm

import (
	"strings"
	"testing"
)

func TestGetPromptCacheFileDefaultsAndCustom(t *testing.T) {
	cacheFile := getPromptCacheFile("qwen3-4b.rkllm", 4096)
	if cacheFile == "" {
		t.Fatal("expected non-empty prompt cache file path")
	}
	if !strings.HasSuffix(cacheFile, ".rkllm_cache") {
		t.Errorf("expected prompt cache file to end with .rkllm_cache, got %q", cacheFile)
	}
}
