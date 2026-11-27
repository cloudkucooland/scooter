package rknn

import (
	"context"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestVisionPipelineSynthetic(t *testing.T) {
	// Test preprocessing and probability normalization
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.RGBA{R: 255, G: 0, B: 0, A: 255})
		}
	}

	nhwc := preprocessImage(img, 224, 224, 3, true)
	if len(nhwc) != 224*224*3 {
		t.Fatalf("expected 224*224*3 bytes, got %d", len(nhwc))
	}
	if nhwc[0] != 255 || nhwc[1] != 0 || nhwc[2] != 0 {
		t.Errorf("unexpected first pixel: %v", nhwc[:3])
	}

	nchw := preprocessImage(img, 224, 224, 3, false)
	if len(nchw) != 224*224*3 {
		t.Fatalf("expected 224*224*3 bytes, got %d", len(nchw))
	}
	if nchw[0] != 255 || nchw[224*224] != 0 || nchw[2*224*224] != 0 {
		t.Errorf("unexpected NCHW red plane pixel: %d, green: %d, blue: %d", nchw[0], nchw[224*224], nchw[2*224*224])
	}

	// Test softmax normalization
	logits := []float32{1.0, 2.0, 3.0}
	probs := normalizeProbabilities(logits)
	if len(probs) != 3 {
		t.Fatalf("expected 3 probabilities, got %d", len(probs))
	}
	sum := probs[0] + probs[1] + probs[2]
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("expected probabilities to sum to 1.0, got %f", sum)
	}
	if probs[2] <= probs[1] || probs[1] <= probs[0] {
		t.Errorf("expected monotonic ascending probabilities: %v", probs)
	}

	// Test summary formatting
	preds := []Prediction{
		{Index: 281, Label: "tabby, tabby cat", Confidence: 0.75},
		{Index: 282, Label: "tiger cat", Confidence: 0.15},
		{Index: 285, Label: "Egyptian cat", Confidence: 0.05},
	}
	summary := formatVisionSummary(preds)
	if !strings.Contains(summary, "tabby, tabby cat") {
		t.Errorf("summary missing top label: %s", summary)
	}
	if !strings.Contains(summary, "75.0%") {
		t.Errorf("summary missing confidence: %s", summary)
	}
}

func TestVisionPipelineHardware(t *testing.T) {
	modelPath := os.Getenv("RKNN_VISION_MODEL")
	if modelPath == "" {
		candidates := []string{
			"/home/scot/3588-stuff/rknn-toolkit2/rknpu2/examples/rknn_mobilenet_demo/model/RK3588/mobilenet_v1.rknn",
			"/home/scot/3588-stuff/rknn-toolkit2/rknn-toolkit-lite2/examples/resnet18/resnet18_for_rk3588.rknn",
		}
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				modelPath = c
				break
			}
		}
	}
	if modelPath == "" {
		t.Skip("no vision model available on host")
	}

	vp, err := LoadVisionPipeline(modelPath)
	if err != nil {
		t.Fatalf("LoadVisionPipeline failed: %v", err)
	}
	defer vp.Close()

	// Check on cat image if available
	testImgPath := "/home/scot/3588-stuff/rknn-toolkit2/rknpu2/examples/rknn_mobilenet_demo/model/cat_224x224.jpg"
	if _, err := os.Stat(testImgPath); err != nil {
		// Generate temporary image
		tmpPath := filepath.Join(t.TempDir(), "test_cat.png")
		img := image.NewRGBA(image.Rect(0, 0, 224, 224))
		f, _ := os.Create(tmpPath)
		_ = png.Encode(f, img)
		_ = f.Close()
		testImgPath = tmpPath
	}

	res, err := vp.DescribeImageFile(context.Background(), testImgPath)
	if err != nil {
		t.Fatalf("DescribeImageFile failed: %v", err)
	}

	t.Logf("Vision Description:\n%s", res.Summary)
	if len(res.Predictions) == 0 {
		t.Error("expected predictions from vision model")
	}
	if res.TopLabel == "" {
		t.Error("expected non-empty TopLabel")
	}
}
