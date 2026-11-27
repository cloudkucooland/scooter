package rknn

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func createMockRKNNModel(t *testing.T, baseDir, modelName string, isSDXL bool) string {
	t.Helper()
	modelDir := filepath.Join(baseDir, modelName)
	submodels := []string{"unet", "text_encoder", "vae_decoder"}
	if isSDXL {
		submodels = append(submodels, "text_encoder_2")
	}
	for _, sub := range submodels {
		subDir := filepath.Join(modelDir, sub)
		if err := os.MkdirAll(subDir, 0755); err != nil {
			t.Fatalf("mkdir %s: %v", subDir, err)
		}
		if err := os.WriteFile(filepath.Join(subDir, "model.rknn"), []byte("dummy rknn binary"), 0644); err != nil {
			t.Fatalf("write dummy model.rknn: %v", err)
		}
	}
	tokDir := filepath.Join(modelDir, "tokenizer")
	if err := os.MkdirAll(tokDir, 0755); err != nil {
		t.Fatalf("mkdir %s: %v", tokDir, err)
	}
	if err := os.WriteFile(filepath.Join(tokDir, "vocab.json"), []byte("{}"), 0644); err != nil {
		t.Fatalf("write dummy vocab: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tokDir, "merges.txt"), []byte("#version: 0.2\n"), 0644); err != nil {
		t.Fatalf("write dummy merges: %v", err)
	}
	return modelDir
}

func TestIsValidRKNNModelDir(t *testing.T) {
	tmp := t.TempDir()

	if IsValidRKNNModelDir(tmp) {
		t.Fatalf("empty dir should not be a valid RKNN model")
	}

	modelDir := createMockRKNNModel(t, tmp, "mock-lcm-1b", false)
	if !IsValidRKNNModelDir(modelDir) {
		t.Fatalf("expected %s to be recognized as a valid RKNN model dir", modelDir)
	}

	// Normalization tests
	if got := NormalizeRKNNModelPath(filepath.Join(modelDir, "unet", "model.rknn")); got != modelDir {
		t.Errorf("NormalizeRKNNModelPath(%s) = %s, want %s", filepath.Join(modelDir, "unet", "model.rknn"), got, modelDir)
	}
	if got := NormalizeRKNNModelPath(filepath.Join(modelDir, "unet")); got != modelDir {
		t.Errorf("NormalizeRKNNModelPath(%s) = %s, want %s", filepath.Join(modelDir, "unet"), got, modelDir)
	}
}

func TestListAndGetRKNNModels(t *testing.T) {
	tmp := t.TempDir()
	origEnv := os.Getenv("SCOOTER_RKNN_MODELS_DIR")
	defer func() {
		os.Setenv("SCOOTER_RKNN_MODELS_DIR", origEnv)
	}()

	os.Setenv("SCOOTER_RKNN_MODELS_DIR", tmp)

	m1 := createMockRKNNModel(t, tmp, "sd15-business", false)
	m2 := createMockRKNNModel(t, tmp, "sdxl-turbo", true)

	models := ListRKNNModels()
	if len(models) < 2 {
		t.Fatalf("expected at least 2 models, got %d: %v", len(models), models)
	}

	foundM1 := false
	foundM2 := false
	for _, m := range models {
		if strings.HasSuffix(m, "sd15-business") {
			foundM1 = true
		}
		if strings.HasSuffix(m, "sdxl-turbo") {
			foundM2 = true
		}
	}
	if !foundM1 || !foundM2 {
		t.Errorf("expected both models in list, got: %v", models)
	}

	// Test GetRKNNModelPath exact base match
	resolvedM1 := GetRKNNModelPath("sd15-business")
	if resolvedM1 != m1 {
		t.Errorf("GetRKNNModelPath('sd15-business') = %s, want %s", resolvedM1, m1)
	}

	// Test GetRKNNModelPath substring match
	resolvedM2 := GetRKNNModelPath("turbo")
	if resolvedM2 != m2 {
		t.Errorf("GetRKNNModelPath('turbo') = %s, want %s", resolvedM2, m2)
	}

	// Test CheckRKNNStatus
	statusM1 := CheckRKNNStatus(m1)
	if !strings.Contains(statusM1, "SD1.5") || !strings.Contains(statusM1, "Model directory valid") {
		t.Errorf("CheckRKNNStatus(m1) = %s, expected SD1.5 and Model directory valid", statusM1)
	}

	statusM2 := CheckRKNNStatus(m2)
	if !strings.Contains(statusM2, "SDXL") || !strings.Contains(statusM2, "text_encoder_2") {
		t.Errorf("CheckRKNNStatus(m2) = %s, expected SDXL and text_encoder_2", statusM2)
	}
}
