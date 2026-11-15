package rknn

import (
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unsafe"
)

func TestCABIStructLayouts(t *testing.T) {
	if got := unsafe.Sizeof(InputOutputNum{}); got != 8 {
		t.Errorf("InputOutputNum size = %d, want 8", got)
	}
	if got := unsafe.Sizeof(Input{}); got != 32 {
		t.Errorf("Input size = %d, want 32", got)
	}
	if got := unsafe.Offsetof(Input{}.Buf); got != 8 {
		t.Errorf("Input.Buf offset = %d, want 8", got)
	}
	if got := unsafe.Sizeof(Output{}); got != 24 {
		t.Errorf("Output size = %d, want 24", got)
	}
	if got := unsafe.Offsetof(Output{}.Index); got != 4 {
		t.Errorf("Output.Index offset = %d, want 4", got)
	}
	if got := unsafe.Offsetof(Output{}.Buf); got != 8 {
		t.Errorf("Output.Buf offset = %d, want 8", got)
	}
	if got := unsafe.Sizeof(TensorAttr{}); got != 376 {
		t.Errorf("TensorAttr size = %d, want 376", got)
	}
	if got := unsafe.Sizeof(TensorMem{}); got != 32 {
		t.Errorf("TensorMem size = %d, want 32", got)
	}
}

func TestSchedulerMath(t *testing.T) {
	lcm := NewScheduler(4, "lcm")
	wantSteps := []int{999, 749, 499, 249}
	for i, want := range wantSteps {
		got, ok := lcm.NextTimestep()
		if !ok || got != want {
			t.Fatalf("step %d: got (%d, %v), want (%d, true)", i, got, ok, want)
		}
		sample := []float32{1.0, -1.0, 0.5, -0.5}
		eps := []float32{0.0, 0.0, 0.0, 0.0}
		next := lcm.Step(eps, got, sample)
		for j, v := range next {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("step %d produced invalid value at %d: %v", i, j, v)
			}
		}
	}
	if _, ok := lcm.NextTimestep(); ok {
		t.Error("expected NextTimestep to return false after 4 steps")
	}
}

func TestCLIPTokenizer(t *testing.T) {
	modelDir := os.Getenv("RKNN_MODEL_DIR")
	if modelDir == "" {
		t.Skip("set RKNN_MODEL_DIR to run CLIP tokenizer test with vocab.json and merges.txt")
	}

	vocabPath := filepath.Join(modelDir, "tokenizer", "vocab.json")
	mergesPath := filepath.Join(modelDir, "tokenizer", "merges.txt")
	tok, err := NewTokenizer(vocabPath, mergesPath)
	if err != nil {
		t.Fatalf("NewTokenizer failed: %v", err)
	}

	ids1, ids2, err := tok.EncodeDual("a photo of a cat")
	if err != nil {
		t.Fatalf("EncodeDual failed: %v", err)
	}

	wantPrefix := []int32{49406, 320, 1125, 539, 320, 2368, 49407}
	for i, want := range wantPrefix {
		if ids1[i] != want {
			t.Errorf("ids1[%d] = %d, want %d", i, ids1[i], want)
		}
		if ids2[i] != want {
			t.Errorf("ids2[%d] = %d, want %d", i, ids2[i], want)
		}
	}
	if ids1[76] != 49407 {
		t.Errorf("ids1[76] pad = %d, want 49407", ids1[76])
	}
	if ids2[76] != 0 {
		t.Errorf("ids2[76] pad = %d, want 0", ids2[76])
	}
}

func TestInspectRKNNModels(t *testing.T) {
	modelDir := os.Getenv("RKNN_MODEL_DIR")
	if modelDir == "" {
		t.Skip("set RKNN_MODEL_DIR to inspect compiled .rknn tensor signatures on NPU")
	}

	submodels := []string{"text_encoder", "text_encoder_2", "unet", "vae_decoder"}
	for _, sub := range submodels {
		path := filepath.Join(modelDir, sub, "model.rknn")
		if _, err := os.Stat(path); err != nil {
			t.Logf("skipping %s: %v", sub, err)
			continue
		}

		var ctx RKNNContext
		if err := ctx.Init(path); err != nil {
			t.Fatalf("%s Init failed: %v", sub, err)
		}
		inputs, outputs, err := ctx.QueryIO()
		_ = ctx.Close()
		if err != nil {
			t.Fatalf("%s QueryIO failed: %v", sub, err)
		}

		t.Logf("=== %s ===", sub)
		for _, in := range inputs {
			name := strings.TrimRight(string(in.Name[:]), "\x00")
			t.Logf("  IN  [%d] %q dims=%v type=%d fmt=%d size=%d", in.Index, name, in.Dims[:in.NDims], in.Type, in.Fmt, in.Size)
		}
		for _, out := range outputs {
			name := strings.TrimRight(string(out.Name[:]), "\x00")
			t.Logf("  OUT [%d] %q dims=%v type=%d fmt=%d size=%d", out.Index, name, out.Dims[:out.NDims], out.Type, out.Fmt, out.Size)
		}
	}
}

func TestGenerateImage(t *testing.T) {
	modelDir := os.Getenv("RKNN_MODEL_DIR")
	if modelDir == "" {
		t.Skip("set RKNN_MODEL_DIR to run end-to-end NPU image generation test")
	}

	tok, err := NewTokenizer(
		filepath.Join(modelDir, "tokenizer", "vocab.json"),
		filepath.Join(modelDir, "tokenizer", "merges.txt"),
	)
	if err != nil {
		t.Fatalf("NewTokenizer failed: %v", err)
	}

	pipe, err := LoadPipeline(modelDir, tok, 4)
	if err != nil {
		t.Fatalf("LoadPipeline failed: %v", err)
	}
	defer pipe.Close()

	img, err := pipe.Generate("a cinematic photo of a red scooter parked on a cobblestone street at sunset", 42)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	outPath := filepath.Join(os.TempDir(), "rknn_test_out.png")
	f, err := os.Create(outPath)
	if err != nil {
		t.Fatalf("create output file: %v", err)
	}
	defer f.Close()

	if err := png.Encode(f, img); err != nil {
		t.Fatalf("png encode: %v", err)
	}
	t.Logf("Saved %dx%d image to %s", img.Bounds().Dx(), img.Bounds().Dy(), outPath)
}
