package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type Config struct {
	DefaultContextTokens int    `json:"default_context_tokens"`
	MinContextTokens     int    `json:"min_context_tokens"`
	MaxContextTokens     int    `json:"max_context_tokens"`
	ContextSafetyTokens  int    `json:"context_safety_tokens"`
	SummaryReserveTokens int    `json:"summary_reserve_tokens"`
	HydrationTokenLimit  int    `json:"hydration_token_limit"`
	DefaultModel         string `json:"default_model"`
	RKLLMModelsDir       string `json:"rkllm_models_dir"`
	RKLLMLib             string `json:"rkllm_lib"`
	MaxGrepFileSize      int64  `json:"max_grep_file_size"`
	MaxShellOutputSize   int    `json:"max_shell_output_size"`
	DefaultToolset       string `json:"default_toolset"`
	GoogleAPIKey         string `json:"google_api_key"`
	GoogleGenAIModel     string `json:"google_genai_model"`
}

var (
	defaultConfig = Config{
		DefaultContextTokens: 4096,
		MinContextTokens:     1024,
		MaxContextTokens:     128 * 1024,
		ContextSafetyTokens:  256,
		SummaryReserveTokens: 512,
		HydrationTokenLimit:  1000,
		DefaultModel:         "qwen3-4b",
		RKLLMLib:             "librkllmrt.so",
		MaxGrepFileSize:      2 * 1024 * 1024,
		MaxShellOutputSize:   64 * 1024,
		DefaultToolset:       "all",
		GoogleGenAIModel:     "gemini-2.5-flash",
	}

	configMu      sync.RWMutex
	cachedCfg     *Config
	cachedPath    string
	cachedModTime time.Time
)

func getConfigFilePath() (string, error) {
	if custom := os.Getenv("SCOOTER_CONFIG_FILE"); custom != "" {
		dir := filepath.Dir(custom)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return "", err
		}
		return custom, nil
	}

	configDir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}

	scooterDir := filepath.Join(configDir, "scooter")
	if customDir := os.Getenv("SCOOTER_CONFIG_DIR"); customDir != "" {
		scooterDir = customDir
	}
	if err := os.MkdirAll(scooterDir, 0755); err != nil {
		return "", err
	}

	return filepath.Join(scooterDir, "config.json"), nil
}

func loadConfig() Config {
	configMu.Lock()
	defer configMu.Unlock()

	path, err := getConfigFilePath()
	var mtime time.Time
	if err == nil {
		if info, statErr := os.Stat(path); statErr == nil {
			mtime = info.ModTime()
		}
	}

	if cachedCfg != nil && cachedPath == path && cachedModTime.Equal(mtime) {
		return *cachedCfg
	}

	cfg := defaultConfig
	if err == nil {
		if data, err := os.ReadFile(path); err == nil {
			temp := defaultConfig
			if err := json.Unmarshal(data, &temp); err == nil {
				cfg = temp
			}
		}
	}

	cachedCfg = &cfg
	cachedPath = path
	cachedModTime = mtime
	return cfg
}

func reloadConfig() Config {
	configMu.Lock()
	cachedCfg = nil
	cachedPath = ""
	cachedModTime = time.Time{}
	configMu.Unlock()
	return loadConfig()
}

func getDefaultContextTokens() int {
	if v := loadConfig().DefaultContextTokens; v > 0 {
		return v
	}
	return defaultConfig.DefaultContextTokens
}

func getMinContextTokens() int {
	if v := loadConfig().MinContextTokens; v > 0 {
		return v
	}
	return defaultConfig.MinContextTokens
}

func getMaxContextTokens() int {
	if v := loadConfig().MaxContextTokens; v > 0 {
		return v
	}
	return defaultConfig.MaxContextTokens
}

func getContextSafetyTokens() int {
	if v := loadConfig().ContextSafetyTokens; v > 0 {
		return v
	}
	return defaultConfig.ContextSafetyTokens
}

func getSummaryReserveTokens() int {
	if v := loadConfig().SummaryReserveTokens; v > 0 {
		return v
	}
	return defaultConfig.SummaryReserveTokens
}

func getHydrationTokenLimit() int {
	if v := loadConfig().HydrationTokenLimit; v > 0 {
		return v
	}
	return defaultConfig.HydrationTokenLimit
}

func getMaxGrepFileSize() int64 {
	if v := loadConfig().MaxGrepFileSize; v > 0 {
		return v
	}
	return defaultConfig.MaxGrepFileSize
}

func getMaxShellOutputSize() int {
	if v := loadConfig().MaxShellOutputSize; v > 0 {
		return v
	}
	return defaultConfig.MaxShellOutputSize
}

func getGoogleAPIKey() string {
	if key := os.Getenv("GOOGLE_API_KEY"); key != "" {
		return key
	}
	return loadConfig().GoogleAPIKey
}

func getGoogleGenAIModel() string {
	if m := os.Getenv("GOOGLE_GENAI_MODEL"); m != "" {
		return m
	}
	if m := os.Getenv("SCOOTER_GENAI_MODEL"); m != "" {
		return m
	}
	if m := loadConfig().GoogleGenAIModel; m != "" {
		return m
	}
	return defaultConfig.GoogleGenAIModel
}
