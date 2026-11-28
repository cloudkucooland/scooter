package rknn

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	sp "github.com/tggo/goSentencePiece"
)

const (
	BGEModelName       = "bge-m3-rknn2-rk3588"
	BGEEmbeddingSize   = 1024
	BGEModelMaxTokens  = 512
	BGEModelVersion    = "f26fcf19bcab5ef685474c38d010596c07eed6d8"
	bgeModelFilename   = "bge-m3_dynamic.rknn"
	bgeTokenizerFile   = "tokenizer.json"
	bgeModelPadTokenID = int64(1)
)

// BGEEmbedder runs Firefly's BGE-M3 RKNN2 model directly on the RK3588 NPU.
// It keeps one model context and tokenizer open for repeated chunk/query calls.
type BGEEmbedder struct {
	mu          sync.Mutex
	ctx         RKNNContext
	tokenizer   *sp.Tokenizer
	inputAttrs  []TensorAttr
	inputIDs    map[string]uint32
	inputValues map[string][]int64
	shapes      []TensorAttr
	inputs      []Input
	outputIDs   []int
	outputID    int
	modelPath   string
	closed      bool
}

// NewBGEEmbedder loads Firefly's bge-m3_dynamic.rknn and tokenizer.json from
// modelDir. Empty modelDir uses SCOOTER_BGE_M3_MODEL_DIR, then
// ~/3588-stuff/rknn-models/bge-m3-rknn2-rk3588.
func NewBGEEmbedder(modelDir string) (*BGEEmbedder, error) {
	modelDir = strings.TrimSpace(modelDir)
	if modelDir == "" {
		modelDir = strings.TrimSpace(os.Getenv("SCOOTER_BGE_M3_MODEL_DIR"))
	}
	if modelDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("resolve home directory for BGE-M3 model: %w", err)
		}
		modelDir = filepath.Join(home, "3588-stuff", "rknn-models", "bge-m3-rknn2-rk3588")
	}
	if strings.EqualFold(filepath.Ext(modelDir), ".rknn") {
		modelDir = filepath.Dir(modelDir)
	}
	modelPath := filepath.Join(modelDir, bgeModelFilename)
	tokenizerPath := filepath.Join(modelDir, bgeTokenizerFile)
	for _, path := range []string{modelPath, tokenizerPath} {
		if info, err := os.Stat(path); err != nil || info.IsDir() {
			if err == nil {
				err = fmt.Errorf("is a directory")
			}
			return nil, fmt.Errorf("open BGE-M3 asset %s: %w", path, err)
		}
	}
	tokenizer, err := sp.NewTokenizerFromJSON(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("load BGE-M3 tokenizer: %w", err)
	}
	if err := LoadLibrary(); err != nil {
		return nil, err
	}
	embedder := &BGEEmbedder{
		tokenizer: tokenizer,
		inputIDs:  make(map[string]uint32, 3),
		inputValues: map[string][]int64{
			"input_ids": {}, "attention_mask": {}, "position_ids": {},
		},
		outputIDs: []int{-1},
		outputID:  -1,
		modelPath: modelPath,
	}
	if err := embedder.ctx.Init(modelPath); err != nil {
		return nil, fmt.Errorf("initialize BGE-M3 RKNN model: %w", err)
	}
	inputs, outputs, err := embedder.ctx.QueryIO()
	if err != nil {
		_ = embedder.ctx.Close()
		return nil, fmt.Errorf("query BGE-M3 tensors: %w", err)
	}
	for _, attr := range inputs {
		name := tensorAttrName(attr)
		switch name {
		case "input_ids", "attention_mask", "position_ids":
			embedder.inputIDs[name] = attr.Index
			if attr.Type != RKNN_TENSOR_INT64 {
				_ = embedder.ctx.Close()
				return nil, fmt.Errorf("BGE-M3 input %q has tensor type %d; expected int64", name, attr.Type)
			}
		}
	}
	for _, name := range []string{"input_ids", "attention_mask", "position_ids"} {
		if _, ok := embedder.inputIDs[name]; !ok {
			_ = embedder.ctx.Close()
			return nil, fmt.Errorf("BGE-M3 model is missing input tensor %q", name)
		}
	}
	for _, attr := range outputs {
		if tensorAttrName(attr) == "sentence_embedding" {
			embedder.outputID = int(attr.Index)
			break
		}
	}
	if embedder.outputID < 0 {
		_ = embedder.ctx.Close()
		return nil, fmt.Errorf("BGE-M3 model is missing sentence_embedding output")
	}
	embedder.inputAttrs = inputs
	return embedder, nil
}

// Model returns the stable model identifier used with stored vectors.
func (e *BGEEmbedder) Model() string { return BGEModelName }

// ModelVersion returns the Hugging Face revision used for this model artifact.
func (e *BGEEmbedder) ModelVersion() string { return BGEModelVersion }

// Embed encodes text into a normalized 1024-dimensional BGE-M3 vector.
func (e *BGEEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("text to embed must not be empty")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	ids, err := e.tokenizer.Encode(text)
	if err != nil {
		return nil, fmt.Errorf("tokenize BGE-M3 input: %w", err)
	}
	// The Firefly dynamic RKNN export supports sequence lengths through 512.
	// Reserve two positions for XLM-RoBERTa's <s> and </s> tokens.
	if len(ids) > BGEModelMaxTokens-2 {
		ids = ids[:BGEModelMaxTokens-2]
	}
	ids = e.tokenizer.AddSpecialTokens(ids)
	sequenceLength := bgeSequenceLength(len(ids))

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, fmt.Errorf("BGE-M3 embedder is closed")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for name, values := range e.inputValues {
		values = values[:sequenceLength]
		for i := range values {
			values[i] = bgeModelPadTokenID
		}
		e.inputValues[name] = values
	}
	inputIDs := e.inputValues["input_ids"]
	attentionMask := e.inputValues["attention_mask"]
	positionIDs := e.inputValues["position_ids"]
	for i, id := range ids {
		inputIDs[i] = int64(id)
		attentionMask[i] = 1
		positionIDs[i] = int64(i) + bgeModelPadTokenID + 1
	}
	e.shapes = append(e.shapes[:0], e.inputAttrs...)
	shapes := e.shapes
	for i := range shapes {
		shapes[i].NDims = 2
		shapes[i].Dims[0] = 1
		shapes[i].Dims[1] = uint32(sequenceLength)
		shapes[i].NElems = uint32(sequenceLength)
		shapes[i].Size = uint32(sequenceLength * 8)
	}
	if err := e.ctx.SetInputShapes(shapes); err != nil {
		return nil, fmt.Errorf("set BGE-M3 sequence length %d: %w", sequenceLength, err)
	}

	e.inputs = e.inputs[:0]
	for _, attr := range shapes {
		name := tensorAttrName(attr)
		values, ok := e.inputValues[name]
		if !ok {
			return nil, fmt.Errorf("unsupported BGE-M3 input tensor %q", name)
		}
		e.inputs = append(e.inputs, Input{
			Index: attr.Index,
			Buf:   unsafe.Pointer(&values[0]),
			Size:  uint32(len(values) * 8),
			Type:  RKNN_TENSOR_INT64,
			Fmt:   attr.Fmt,
		})
	}
	e.outputIDs[0] = e.outputID
	outputs, err := e.ctx.RunFloat32Indices(e.inputs, e.outputIDs)
	runtime.KeepAlive(inputIDs)
	runtime.KeepAlive(attentionMask)
	runtime.KeepAlive(positionIDs)
	if err != nil {
		return nil, fmt.Errorf("run BGE-M3 inference: %w", err)
	}
	if len(outputs) != 1 || len(outputs[0]) != BGEEmbeddingSize {
		got := 0
		if len(outputs) > 0 {
			got = len(outputs[0])
		}
		return nil, fmt.Errorf("BGE-M3 returned %d embedding values; expected %d", got, BGEEmbeddingSize)
	}
	return normalizeEmbedding(outputs[0])
}

// Close releases the model context and its NPU memory.
func (e *BGEEmbedder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	return e.ctx.Close()
}

func tensorAttrName(attr TensorAttr) string {
	name := string(attr.Name[:])
	return strings.TrimRight(name, "\x00")
}

func bgeSequenceLength(tokenCount int) int {
	length := 8
	for length < tokenCount && length < BGEModelMaxTokens {
		length *= 2
	}
	return length
}

func normalizeEmbedding(vector []float32) ([]float32, error) {
	var normSquared float64
	for i, value := range vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("BGE-M3 output contains a non-finite value at index %d", i)
		}
		normSquared += float64(value) * float64(value)
	}
	if normSquared == 0 || math.IsInf(normSquared, 0) {
		return nil, fmt.Errorf("BGE-M3 output has invalid vector norm")
	}
	inverseNorm := float32(1 / math.Sqrt(normSquared))
	normalized := make([]float32, len(vector))
	for i, value := range vector {
		normalized[i] = value * inverseNorm
	}
	return normalized, nil
}
