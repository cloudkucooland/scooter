package main

import (
	"context"
	"fmt"
	"image/png"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"codeberg.org/indievisible/scooter/pkg/rkllm"
	"codeberg.org/indievisible/scooter/pkg/rknn"
)

var (
	sharedCLIPTok     *rknn.Tokenizer
	sharedCLIPTokDir  string
	sharedCLIPTokOnce sync.Mutex
)

const defaultCFGNegativePrompt = "deformed, distorted, disfigured, extra limbs, extra spouts, symmetrical, duplicate, mutated, blurry, low quality, bad anatomy"

func getCLIPTokenizer(modelDir string) (*rknn.Tokenizer, error) {
	sharedCLIPTokOnce.Lock()
	defer sharedCLIPTokOnce.Unlock()

	if sharedCLIPTok != nil && sharedCLIPTokDir == modelDir {
		return sharedCLIPTok, nil
	}
	vocabPath := filepath.Join(modelDir, "tokenizer", "vocab.json")
	mergesPath := filepath.Join(modelDir, "tokenizer", "merges.txt")
	tok, err := rknn.NewTokenizer(vocabPath, mergesPath)
	if err != nil {
		return nil, err
	}
	sharedCLIPTok = tok
	sharedCLIPTokDir = modelDir
	return tok, nil
}

func generateImageFile(prompt, negativePrompt, outputPath string, steps int, seed int64, guidanceScale float32) (string, time.Duration, error) {
	return generateImageFileWithContext(context.Background(), prompt, negativePrompt, outputPath, steps, seed, guidanceScale, nil)
}

func generateImageFileForModel(modelName, prompt, negativePrompt, outputPath string, steps int, seed int64, guidanceScale float32) (string, time.Duration, error) {
	targetModel := strings.TrimSpace(modelName)
	if targetModel == "" {
		targetModel = getActiveRKNNModelPath()
	} else {
		targetModel = rknn.GetRKNNModelPath(targetModel)
	}
	return generateImageFileWithModelAndContext(context.Background(), targetModel, prompt, negativePrompt, outputPath, steps, seed, guidanceScale, nil)
}

func generateImageFileWithContext(ctx context.Context, prompt, negativePrompt, outputPath string, steps int, seed int64, guidanceScale float32, progressFn func(step, total int)) (string, time.Duration, error) {
	return generateImageFileWithModelAndContext(ctx, getActiveRKNNModelPath(), prompt, negativePrompt, outputPath, steps, seed, guidanceScale, progressFn)
}

func generateImageFileWithModelAndContext(ctx context.Context, modelDir, prompt, negativePrompt, outputPath string, steps int, seed int64, guidanceScale float32, progressFn func(step, total int)) (string, time.Duration, error) {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return "", 0, fmt.Errorf("prompt is required")
	}

	if steps <= 0 {
		steps = 4
	}
	if steps > 30 {
		steps = 30
	}

	if seed == 0 {
		rng := rand.New(rand.NewSource(time.Now().UnixNano()))
		seed = rng.Int63n(1<<31 - 1)
	}

	if strings.TrimSpace(outputPath) == "" {
		outputPath = fmt.Sprintf("scooter_%s.png", time.Now().Format("20060102_150405"))
	} else if strings.ToLower(filepath.Ext(outputPath)) != ".png" {
		outputPath += ".png"
	}

	absOutPath, err := validatePath(outputPath)
	if err != nil {
		return "", 0, err
	}

	syncRKNNEnv()
	if strings.TrimSpace(modelDir) == "" {
		modelDir = getActiveRKNNModelPath()
	} else {
		modelDir = rknn.GetRKNNModelPath(modelDir)
	}
	tok, err := getCLIPTokenizer(modelDir)
	if err != nil {
		return "", 0, fmt.Errorf("load CLIP tokenizer from %s: %w", modelDir, err)
	}

	start := time.Now()

	// Release active RKLLM CMA memory before loading UNet/VAE
	rkllm.CloseRKLLMSession()

	pipe, err := rknn.LoadPipeline(modelDir, tok, steps)
	if err != nil {
		return "", 0, fmt.Errorf("load RKNN pipeline from %s: %w", modelDir, err)
	}
	defer pipe.Close()

	if guidanceScale > 0 {
		if pipe.UseCFG {
			if guidanceScale < 1.0 {
				guidanceScale = 1.0
			} else if guidanceScale > 15.0 {
				guidanceScale = 15.0
			}
		} else {
			if guidanceScale < 1.0 {
				guidanceScale = 1.0
			} else if guidanceScale > 2.0 {
				guidanceScale = 2.0
			}
		}
		pipe.GuidanceScale = guidanceScale
	}

	effectiveGuidance := pipe.GuidanceScale
	if effectiveGuidance == 0 {
		if pipe.UseCFG {
			effectiveGuidance = 7.5
		} else {
			effectiveGuidance = 1.5
		}
	}

	neg := strings.TrimSpace(negativePrompt)
	if strings.EqualFold(neg, "none") {
		neg = ""
	} else if neg == "" && pipe.UseCFG {
		neg = defaultCFGNegativePrompt
	}

	img, err := pipe.GenerateWithNegative(ctx, prompt, neg, seed, progressFn)
	if err != nil {
		return "", 0, err
	}

	outDir := filepath.Dir(absOutPath)
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return "", 0, fmt.Errorf("create output directory %s: %w", outDir, err)
	}

	tmpFile, err := os.CreateTemp(outDir, "scooter-img-*.png")
	if err != nil {
		return "", 0, fmt.Errorf("create temp image file: %w", err)
	}
	tmpName := tmpFile.Name()
	defer func() {
		_ = tmpFile.Close()
		_ = os.Remove(tmpName)
	}()

	if err := png.Encode(tmpFile, img); err != nil {
		return "", 0, fmt.Errorf("encode PNG: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		return "", 0, fmt.Errorf("close temp PNG: %w", err)
	}
	if err := os.Rename(tmpName, absOutPath); err != nil {
		return "", 0, fmt.Errorf("save PNG to %s: %w", absOutPath, err)
	}

	elapsed := time.Since(start)
	relOut := absOutPath
	if workingDir != "" {
		if rel, relErr := filepath.Rel(workingDir, absOutPath); relErr == nil {
			relOut = rel
		}
	}
	return fmt.Sprintf("%s (%dx%d, %d steps, guidance %.1f, seed %d)", relOut, img.Bounds().Dx(), img.Bounds().Dy(), steps, effectiveGuidance, seed), elapsed, nil
}
