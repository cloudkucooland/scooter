package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"codeberg.org/indievisible/scooter/pkg/rknn"
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
	RKLLMCacheEnabled    bool   `json:"rkllm_cache_enabled"`
	RKLLMCachePath       string `json:"rkllm_cache_path"`
	DefaultRKNNModel     string `json:"default_rknn_model"`
	RKNNModelsDir        string `json:"rknn_models_dir"`
	RKNNLib              string `json:"rknn_lib"`
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
		RKLLMCacheEnabled:    true,
		DefaultRKNNModel:     "lcm-ssd-1b-rknn",
		RKNNLib:              "/usr/lib/librknnrt.so",
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

func saveConfig(cfg Config) error {
	configMu.Lock()
	defer configMu.Unlock()

	path, err := getConfigFilePath()
	if err != nil {
		return err
	}

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return err
	}

	cachedCfg = &cfg
	cachedPath = path
	if info, statErr := os.Stat(path); statErr == nil {
		cachedModTime = info.ModTime()
	}
	return nil
}

func init() {
	rknn.GetConfigRKNNModelsDir = func() string {
		return loadConfig().RKNNModelsDir
	}
}

var (
	activeRKNNModelMu sync.RWMutex
	activeRKNNModel   string
)

func getActiveRKNNModel() string {
	activeRKNNModelMu.RLock()
	defer activeRKNNModelMu.RUnlock()
	if activeRKNNModel != "" {
		return activeRKNNModel
	}
	if env := strings.TrimSpace(os.Getenv("SCOOTER_RKNN_MODEL")); env != "" {
		return env
	}
	if env := strings.TrimSpace(os.Getenv("RKNN_MODEL")); env != "" {
		return env
	}
	if def := strings.TrimSpace(loadConfig().DefaultRKNNModel); def != "" {
		return def
	}
	return "lcm-ssd-1b-rknn"
}

func setActiveRKNNModel(m string) {
	activeRKNNModelMu.Lock()
	defer activeRKNNModelMu.Unlock()
	activeRKNNModel = m
}

func getActiveRKNNModelPath() string {
	return rknn.GetRKNNModelPath(getActiveRKNNModel())
}

func getRKNNModelsDir() string {
	for _, envVar := range []string{"SCOOTER_RKNN_MODELS_DIR", "RKNN_MODEL_DIR"} {
		if dir := strings.TrimSpace(os.Getenv(envVar)); dir != "" {
			return dir
		}
	}
	if dir := strings.TrimSpace(loadConfig().RKNNModelsDir); dir != "" {
		return dir
	}

	candidates := []string{
		"lcm-ssd-1b-rknn",
	}
	if homeDir, err := os.UserHomeDir(); err == nil {
		candidates = append(candidates,
			filepath.Join(homeDir, "3588-stuff", "rknn-models", "lcm-ssd-1b-rknn"),
			filepath.Join(homeDir, "rkllama-models", "rknn-models", "lcm-ssd-1b-rknn"),
		)
	}
	for _, candidate := range candidates {
		if info, err := os.Stat(filepath.Join(candidate, "unet", "model.rknn")); err == nil && !info.IsDir() {
			if abs, absErr := filepath.Abs(candidate); absErr == nil {
				return abs
			}
			return candidate
		}
	}
	return filepath.Join("pkg", "rknn", "lcm-ssd-1b-rknn")
}

func syncRKNNEnv() {
	if os.Getenv("RKNN_LIB") == "" {
		if lib := strings.TrimSpace(loadConfig().RKNNLib); lib != "" {
			_ = os.Setenv("RKNN_LIB", lib)
		}
	}
}
