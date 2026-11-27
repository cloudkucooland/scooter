package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"codeberg.org/indievisible/scooter/pkg/rkllm"
	"codeberg.org/indievisible/scooter/pkg/rknn"
)

func describeImageFile(imagePath, modelName string) (string, time.Duration, error) {
	return describeImageFileWithContext(context.Background(), imagePath, modelName)
}

func describeImageFileWithContext(ctx context.Context, imagePath, modelName string) (string, time.Duration, error) {
	cleanPath := strings.TrimSpace(imagePath)
	if cleanPath == "" {
		return "", 0, fmt.Errorf("image_path is required")
	}

	targetPath, err := validatePath(cleanPath)
	if err != nil {
		if info, statErr := os.Stat(cleanPath); statErr == nil && !info.IsDir() {
			if abs, absErr := filepath.Abs(cleanPath); absErr == nil {
				targetPath = abs
			} else {
				targetPath = cleanPath
			}
		} else {
			return "", 0, err
		}
	}

	syncRKNNEnv()

	// Release active RKLLM CMA memory before loading vision model
	rkllm.CloseRKLLMSession()

	start := time.Now()

	resolvedModel := strings.TrimSpace(modelName)
	if resolvedModel == "" {
		resolvedModel = getActiveRKNNVisionModel()
	}

	pipe, err := rknn.LoadVisionPipeline(resolvedModel)
	if err != nil {
		return "", 0, fmt.Errorf("load RKNN vision model: %w", err)
	}
	defer pipe.Close()

	res, err := pipe.DescribeImageFile(ctx, targetPath)
	if err != nil {
		return "", 0, fmt.Errorf("describe image %s: %w", targetPath, err)
	}

	elapsed := time.Since(start)

	relOut := targetPath
	if workingDir != "" {
		if rel, relErr := filepath.Rel(workingDir, targetPath); relErr == nil && !strings.HasPrefix(rel, "..") {
			relOut = rel
		}
	}

	resultText := fmt.Sprintf("%s (%dx%d, model: %s)\n%s", relOut, res.ImageWidth, res.ImageHeight, filepath.Base(res.ModelPath), res.Summary)
	return resultText, elapsed, nil
}
