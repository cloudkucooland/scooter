package rknn

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

var (
	// GetConfigRKNNModelsDir can be provided by caller to supply configured models directory
	GetConfigRKNNModelsDir func() string
)

// IsValidRKNNModelDir checks whether a given directory qualifies as an RKNN model directory.
// A valid RKNN model directory contains at minimum unet/model.rknn.
func IsValidRKNNModelDir(path string) bool {
	if path == "" {
		return false
	}
	info, err := os.Stat(path)
	if err != nil || !info.IsDir() {
		return false
	}
	unetPath := filepath.Join(path, "unet", "model.rknn")
	if uInfo, err := os.Stat(unetPath); err == nil && !uInfo.IsDir() {
		return true
	}
	return false
}

// NormalizeRKNNModelPath normalizes paths that may point to submodels (like unet or unet/model.rknn)
// back to the root model directory if valid.
func NormalizeRKNNModelPath(path string) string {
	clean := strings.TrimSpace(path)
	if clean == "" {
		return ""
	}
	sep := string(filepath.Separator)
	if strings.HasSuffix(clean, "/unet/model.rknn") || strings.HasSuffix(clean, sep+"unet"+sep+"model.rknn") {
		parent := filepath.Dir(filepath.Dir(clean))
		if IsValidRKNNModelDir(parent) {
			return parent
		}
	}
	if strings.HasSuffix(clean, "/unet") || strings.HasSuffix(clean, sep+"unet") {
		parent := filepath.Dir(clean)
		if IsValidRKNNModelDir(parent) {
			return parent
		}
	}
	return clean
}

// ListRKNNModels scans standard and configured directories for available RKNN models.
func ListRKNNModels() []string {
	var models []string
	seen := make(map[string]bool)

	addModel := func(path string) {
		if path == "" {
			return
		}
		normalized := NormalizeRKNNModelPath(path)
		abs, err := filepath.Abs(normalized)
		if err == nil {
			normalized = abs
		}
		if !seen[normalized] {
			seen[normalized] = true
			models = append(models, normalized)
		}
	}

	// 1. Direct environment variables
	for _, envVar := range []string{"SCOOTER_RKNN_MODEL", "RKNN_MODEL", "RKNN_MODEL_DIR"} {
		if val := strings.TrimSpace(os.Getenv(envVar)); val != "" {
			norm := NormalizeRKNNModelPath(val)
			if IsValidRKNNModelDir(norm) {
				addModel(norm)
			}
		}
	}

	// 2. Collect root search directories
	var roots []string
	for _, envVar := range []string{"SCOOTER_RKNN_MODELS_DIR", "RKNN_MODELS_DIR", "RKNN_MODEL_DIR"} {
		if val := strings.TrimSpace(os.Getenv(envVar)); val != "" {
			roots = append(roots, val)
		}
	}
	if GetConfigRKNNModelsDir != nil {
		if cfgDir := strings.TrimSpace(GetConfigRKNNModelsDir()); cfgDir != "" {
			roots = append(roots, cfgDir)
		}
	}

	if homeDir, err := os.UserHomeDir(); err == nil {
		roots = append(roots,
			filepath.Join(homeDir, "3588-stuff", "rknn-models"),
			filepath.Join(homeDir, "rkllama-models", "rknn-models"),
			filepath.Join(homeDir, "rknn-models"),
			filepath.Join(homeDir, "3588-stuff"),
		)
	}
	roots = append(roots,
		"rknn-models",
		filepath.Join("pkg", "rknn"),
		"lcm-ssd-1b-rknn",
		".",
	)

	// 3. Scan roots
	for _, root := range roots {
		info, err := os.Stat(root)
		if err != nil || !info.IsDir() {
			continue
		}
		if IsValidRKNNModelDir(root) {
			addModel(root)
			continue
		}
		_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			if !d.IsDir() {
				return nil
			}
			if p != root && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			if p != root && IsValidRKNNModelDir(p) {
				addModel(p)
				return filepath.SkipDir
			}
			rel, relErr := filepath.Rel(root, p)
			if relErr == nil && strings.Count(rel, string(filepath.Separator)) >= 4 {
				return filepath.SkipDir
			}
			return nil
		})
	}

	if len(models) == 0 {
		models = append(models, "No RKNN models found")
	}
	return models
}

// GetDefaultRKNNModelName returns default RKNN model path or identifier
func GetDefaultRKNNModelName() string {
	for _, env := range []string{"SCOOTER_RKNN_MODEL", "RKNN_MODEL"} {
		if val := strings.TrimSpace(os.Getenv(env)); val != "" {
			return GetRKNNModelPath(val)
		}
	}
	return GetRKNNModelPath("lcm-ssd-1b-rknn")
}

// GetRKNNModelPath resolves a model name or partial/full path to an RKNN model directory.
func GetRKNNModelPath(modelName string) string {
	clean := strings.TrimSpace(modelName)
	if clean != "" {
		normalized := NormalizeRKNNModelPath(clean)
		if IsValidRKNNModelDir(normalized) {
			if abs, err := filepath.Abs(normalized); err == nil {
				return abs
			}
			return normalized
		}
		if filepath.IsAbs(normalized) || strings.Contains(normalized, string(filepath.Separator)) {
			if _, err := os.Stat(normalized); err == nil {
				return normalized
			}
		}
	}

	if clean == "" {
		for _, env := range []string{"SCOOTER_RKNN_MODEL", "RKNN_MODEL"} {
			if val := strings.TrimSpace(os.Getenv(env)); val != "" {
				norm := NormalizeRKNNModelPath(val)
				if IsValidRKNNModelDir(norm) {
					if abs, err := filepath.Abs(norm); err == nil {
						return abs
					}
					return norm
				}
			}
		}
	}

	models := ListRKNNModels()
	target := strings.ToLower(clean)

	if target != "" {
		for _, m := range models {
			if strings.HasPrefix(m, "No ") {
				continue
			}
			if strings.ToLower(filepath.Base(m)) == target {
				return m
			}
		}
		for _, m := range models {
			if strings.HasPrefix(m, "No ") {
				continue
			}
			if strings.Contains(strings.ToLower(m), target) {
				return m
			}
		}
	}

	for _, m := range models {
		if !strings.HasPrefix(m, "No ") && IsValidRKNNModelDir(m) {
			return m
		}
	}

	if clean != "" {
		return clean
	}
	return "lcm-ssd-1b-rknn"
}

// CheckRKNNStatus inspects the active RKNN model directory and runtime library.
func CheckRKNNStatus(modelDir string) string {
	resolved := GetRKNNModelPath(modelDir)
	rknnLib := os.Getenv("RKNN_LIB")
	if rknnLib == "" {
		rknnLib = "/usr/lib/librknnrt.so"
	}

	status := "Model directory valid"
	if !IsValidRKNNModelDir(resolved) {
		status = "Model directory invalid or missing required unet/model.rknn"
	}

	var submodels []string
	for _, sub := range []string{"text_encoder", "text_encoder_2", "unet", "vae_decoder"} {
		p := filepath.Join(resolved, sub, "model.rknn")
		if _, err := os.Stat(p); err == nil {
			submodels = append(submodels, sub)
		}
	}

	arch := "Unknown / Incomplete"
	if IsValidRKNNModelDir(resolved) {
		if _, err := os.Stat(filepath.Join(resolved, "text_encoder_2", "model.rknn")); err == nil {
			arch = "SDXL"
		} else {
			arch = "SD1.5 / LCM"
		}
	}

	res := fmt.Sprintf("RKNN runtime (lib: %s, model: %s, architecture: %s, status: %s, submodels: [%s])",
		rknnLib, resolved, arch, status, strings.Join(submodels, ", "))
	return res
}
