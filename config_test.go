package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestConfigDefaults(t *testing.T) {
	orig := os.Getenv("SCOOTER_CONFIG_FILE")
	defer func() {
		os.Setenv("SCOOTER_CONFIG_FILE", orig)
		reloadConfig()
	}()
	os.Setenv("SCOOTER_CONFIG_FILE", "/nonexistent/path/config.json")
	reloadConfig()

	cfg := loadConfig()
	if cfg.DefaultContextTokens != 4096 {
		t.Fatalf("expected DefaultContextTokens 4096, got %d", cfg.DefaultContextTokens)
	}
	if cfg.DefaultModel != "qwen3-4b" {
		t.Fatalf("expected DefaultModel 'qwen3-4b', got %q", cfg.DefaultModel)
	}
}

func TestConfigCustomFile(t *testing.T) {
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "config.json")

	customCfg := Config{
		DefaultContextTokens: 8192,
		DefaultModel:         "deepseek-8k",
		MaxGrepFileSize:      1024,
	}
	data, _ := json.Marshal(customCfg)
	if err := os.WriteFile(configFile, data, 0644); err != nil {
		t.Fatal(err)
	}

	orig := os.Getenv("SCOOTER_CONFIG_FILE")
	defer func() {
		os.Setenv("SCOOTER_CONFIG_FILE", orig)
		reloadConfig()
	}()
	os.Setenv("SCOOTER_CONFIG_FILE", configFile)
	reloadConfig()

	cfg := loadConfig()
	if cfg.DefaultContextTokens != 8192 {
		t.Fatalf("expected DefaultContextTokens 8192, got %d", cfg.DefaultContextTokens)
	}
	if cfg.DefaultModel != "deepseek-8k" {
		t.Fatalf("expected DefaultModel 'deepseek-8k', got %q", cfg.DefaultModel)
	}
	if cfg.MaxGrepFileSize != 1024 {
		t.Fatalf("expected MaxGrepFileSize 1024, got %d", cfg.MaxGrepFileSize)
	}
}
