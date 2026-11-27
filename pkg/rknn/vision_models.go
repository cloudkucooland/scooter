package rknn

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// IsValidVisionModelFile checks if the given file exists and is an RKNN model file.
func IsValidVisionModelFile(path string) bool {
	if path == "" {
		return false
	}
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return false
	}
	return strings.HasSuffix(strings.ToLower(path), ".rknn")
}

// ListRKNNVisionModels finds all available vision RKNN models.
func ListRKNNVisionModels() []string {
	var models []string
	seen := make(map[string]bool)

	addModel := func(p string) {
		clean := strings.TrimSpace(p)
		if clean == "" {
			return
		}
		if abs, err := filepath.Abs(clean); err == nil {
			clean = abs
		}
		if !seen[clean] && IsValidVisionModelFile(clean) {
			seen[clean] = true
			models = append(models, clean)
		}
	}

	// 1. Direct env vars
	for _, env := range []string{"SCOOTER_RKNN_VISION_MODEL", "RKNN_VISION_MODEL"} {
		if val := os.Getenv(env); val != "" {
			addModel(val)
		}
	}

	// 2. Known standard paths
	if homeDir, err := os.UserHomeDir(); err == nil {
		knownDemos := []string{
			filepath.Join(homeDir, "3588-stuff", "rknn-toolkit2", "rknpu2", "examples", "rknn_mobilenet_demo", "model", "RK3588", "mobilenet_v1.rknn"),
			filepath.Join(homeDir, "3588-stuff", "rknn-toolkit2", "rknn-toolkit-lite2", "examples", "resnet18", "resnet18_for_rk3588.rknn"),
			filepath.Join(homeDir, "3588-stuff", "rknn-toolkit2", "rknpu2", "examples", "rknn_yolov5_demo", "model", "RK3588", "yolov5s-640-640.rknn"),
		}
		for _, kd := range knownDemos {
			addModel(kd)
		}
	}

	// 3. Search directories
	var roots []string
	for _, env := range []string{"SCOOTER_RKNN_MODELS_DIR", "RKNN_MODELS_DIR", "RKNN_MODEL_DIR"} {
		if val := strings.TrimSpace(os.Getenv(env)); val != "" {
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
			filepath.Join(homeDir, "3588-stuff"),
			filepath.Join(homeDir, "rknn-models"),
			filepath.Join(homeDir, "rkllama-models"),
		)
	}
	roots = append(roots, "rknn-models", "models", ".")

	for _, root := range roots {
		info, err := os.Stat(root)
		if err != nil || !info.IsDir() {
			continue
		}
		_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil || d.IsDir() {
				return nil
			}
			name := strings.ToLower(d.Name())
			if strings.HasSuffix(name, ".rknn") {
				// Skip diffusion submodels
				dirName := strings.ToLower(filepath.Base(filepath.Dir(p)))
				if dirName == "unet" || dirName == "vae_decoder" || dirName == "vae_encoder" || dirName == "text_encoder" || dirName == "text_encoder_2" {
					return nil
				}
				addModel(p)
			}
			return nil
		})
	}

	return models
}

// GetRKNNVisionModelPath resolves a model name or path to an active vision model.
func GetRKNNVisionModelPath(modelName string) string {
	clean := strings.TrimSpace(modelName)
	if clean != "" {
		if IsValidVisionModelFile(clean) {
			if abs, err := filepath.Abs(clean); err == nil {
				return abs
			}
			return clean
		}
	}

	for _, env := range []string{"SCOOTER_RKNN_VISION_MODEL", "RKNN_VISION_MODEL"} {
		if val := strings.TrimSpace(os.Getenv(env)); val != "" {
			if IsValidVisionModelFile(val) {
				if abs, err := filepath.Abs(val); err == nil {
					return abs
				}
				return val
			}
		}
	}

	models := ListRKNNVisionModels()
	if clean != "" {
		target := strings.ToLower(clean)
		for _, m := range models {
			if strings.ToLower(filepath.Base(m)) == target || strings.ToLower(filepath.Base(m)) == target+".rknn" {
				return m
			}
		}
		for _, m := range models {
			if strings.Contains(strings.ToLower(m), target) {
				return m
			}
		}
	}

	if len(models) > 0 {
		return models[0]
	}

	if clean != "" {
		return clean
	}
	return "mobilenet_v1.rknn"
}
