package rknn

import (
	"bufio"
	"context"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unsafe"

	"codeberg.org/indievisible/scooter/pkg/npu"
)

type Prediction struct {
	Index      int     `json:"index"`
	Label      string  `json:"label"`
	Confidence float32 `json:"confidence"`
}

type VisionResult struct {
	ModelPath   string       `json:"model_path"`
	ImageWidth  int          `json:"image_width"`
	ImageHeight int          `json:"image_height"`
	ModelWidth  int          `json:"model_width"`
	ModelHeight int          `json:"model_height"`
	TopLabel    string       `json:"top_label"`
	TopScore    float32      `json:"top_score"`
	Predictions []Prediction `json:"predictions"`
	Summary     string       `json:"summary"`
}

type VisionPipeline struct {
	ModelPath  string
	ctx        *RKNNContext
	inputAttr  TensorAttr
	outputAttr TensorAttr
	Width      int
	Height     int
	Channels   int
	IsNHWC     bool
	Labels     []string
}

// LoadVisionPipeline loads an RKNN vision model (e.g., MobileNet, ResNet, YOLO)
// and prepares it for image classification or description on the NPU.
func LoadVisionPipeline(modelPath string) (*VisionPipeline, error) {
	resolvedPath := GetRKNNVisionModelPath(modelPath)
	if resolvedPath == "" {
		return nil, fmt.Errorf("no RKNN vision model found (specified: %q)", modelPath)
	}

	ctx := &RKNNContext{}
	if err := ctx.Init(resolvedPath); err != nil {
		return nil, fmt.Errorf("init RKNN vision model %s: %w", resolvedPath, err)
	}

	if err := ctx.SetCoreMask(RKNN_NPU_CORE_0_1_2); err != nil {
		_ = ctx.Close()
		return nil, fmt.Errorf("set core mask for vision model: %w", err)
	}

	inputs, outputs, err := ctx.QueryIO()
	if err != nil {
		_ = ctx.Close()
		return nil, fmt.Errorf("query vision model IO: %w", err)
	}
	if len(inputs) == 0 {
		_ = ctx.Close()
		return nil, fmt.Errorf("vision model %s has no inputs", resolvedPath)
	}
	if len(outputs) == 0 {
		_ = ctx.Close()
		return nil, fmt.Errorf("vision model %s has no outputs", resolvedPath)
	}

	in := inputs[0]
	isNHWC := in.Fmt == RKNN_TENSOR_NHWC
	var w, h, c int
	if isNHWC {
		h = int(in.Dims[1])
		w = int(in.Dims[2])
		c = int(in.Dims[3])
	} else {
		c = int(in.Dims[1])
		h = int(in.Dims[2])
		w = int(in.Dims[3])
	}

	if w <= 0 || h <= 0 {
		w, h, c = 224, 224, 3
	}
	if c <= 0 {
		c = 3
	}

	vp := &VisionPipeline{
		ModelPath:  resolvedPath,
		ctx:        ctx,
		inputAttr:  in,
		outputAttr: outputs[0],
		Width:      w,
		Height:     h,
		Channels:   c,
		IsNHWC:     isNHWC,
	}

	vp.loadLabels()
	return vp, nil
}

func (vp *VisionPipeline) Close() error {
	if vp.ctx != nil {
		err := vp.ctx.Close()
		vp.ctx = nil
		return err
	}
	return nil
}

func (vp *VisionPipeline) loadLabels() {
	dir := filepath.Dir(vp.ModelPath)
	base := strings.TrimSuffix(filepath.Base(vp.ModelPath), filepath.Ext(vp.ModelPath))

	candidateFiles := []string{
		filepath.Join(dir, "labels.txt"),
		filepath.Join(dir, "synset.txt"),
		filepath.Join(dir, "classes.txt"),
		filepath.Join(dir, "coco_80_labels_list.txt"),
		filepath.Join(dir, base+"_labels.txt"),
		filepath.Join(dir, base+".txt"),
	}

	for _, cand := range candidateFiles {
		if lines, err := readLines(cand); err == nil && len(lines) > 0 {
			vp.Labels = lines
			return
		}
	}

	outElems := int(vp.outputAttr.NElems)
	if outElems == 0 && vp.outputAttr.Size > 0 {
		outElems = int(vp.outputAttr.Size / 4)
	}

	if outElems == 1000 || outElems == 1001 {
		vp.Labels = make([]string, len(ImageNetLabels))
		copy(vp.Labels, ImageNetLabels[:])
		return
	}

	if outElems == 80 {
		vp.Labels = make([]string, len(COCOLabels))
		copy(vp.Labels, COCOLabels[:])
		return
	}

	vp.Labels = make([]string, outElems)
	for i := range vp.Labels {
		vp.Labels[i] = fmt.Sprintf("class_%d", i)
	}
}

func readLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var lines []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		text := strings.TrimSpace(scanner.Text())
		if text != "" {
			lines = append(lines, text)
		}
	}
	return lines, scanner.Err()
}

// DescribeImageFile reads an image file from disk and generates a text description.
func (vp *VisionPipeline) DescribeImageFile(ctx context.Context, imagePath string) (*VisionResult, error) {
	f, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("open image file %s: %w", imagePath, err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decode image %s: %w", imagePath, err)
	}

	return vp.Describe(ctx, img)
}

// Describe processes an image.Image and runs inference on the NPU to produce a description.
func (vp *VisionPipeline) Describe(ctx context.Context, img image.Image) (*VisionResult, error) {
	if vp.ctx == nil {
		return nil, fmt.Errorf("vision pipeline closed")
	}

	origBounds := img.Bounds()
	origW := origBounds.Dx()
	origH := origBounds.Dy()

	inputBuf := preprocessImage(img, vp.Width, vp.Height, vp.Channels, vp.IsNHWC)

	var input Input
	input.Index = 0
	input.PassThrough = 0

	if vp.inputAttr.Type == RKNN_TENSOR_FLOAT32 {
		f32Buf := make([]float32, len(inputBuf))
		for i, b := range inputBuf {
			f32Buf[i] = float32(b) / 255.0
		}
		input.Buf = unsafe.Pointer(&f32Buf[0])
		input.Size = uint32(len(f32Buf) * 4)
		input.Type = RKNN_TENSOR_FLOAT32
		if vp.IsNHWC {
			input.Fmt = RKNN_TENSOR_NHWC
		} else {
			input.Fmt = RKNN_TENSOR_NCHW
		}
	} else {
		input.Buf = unsafe.Pointer(&inputBuf[0])
		input.Size = uint32(len(inputBuf))
		input.Type = RKNN_TENSOR_UINT8
		if vp.IsNHWC {
			input.Fmt = RKNN_TENSOR_NHWC
		} else {
			input.Fmt = RKNN_TENSOR_NCHW
		}
	}

	npu.Lock()
	defer npu.Unlock()

	if ret := rknnInputsSet(vp.ctx.handle, 1, &input); ret != 0 {
		return nil, fmt.Errorf("rknn_inputs_set failed: %d", ret)
	}

	if ret := rknnRun(vp.ctx.handle, nil); ret != 0 {
		return nil, fmt.Errorf("rknn_run failed: %d", ret)
	}

	out := &Output{
		WantFloat:  1,
		IsPrealloc: 0,
		Index:      0,
	}

	if ret := callRKNNOutputsGet(vp.ctx.handle, 1, out); ret != 0 {
		return nil, fmt.Errorf("rknn_outputs_get failed: %d", ret)
	}
	defer callRKNNOutputsRelease(vp.ctx.handle, 1, out)

	if out.Buf == nil || out.Size == 0 {
		return nil, fmt.Errorf("rknn_outputs_get returned empty buffer")
	}

	nFloats := int(out.Size / 4)
	rawOutputs := unsafe.Slice((*float32)(out.Buf), nFloats)
	outputCopy := make([]float32, nFloats)
	copy(outputCopy, rawOutputs)

	probs := normalizeProbabilities(outputCopy)

	type scoredIdx struct {
		idx   int
		score float32
	}
	scored := make([]scoredIdx, len(probs))
	for i, p := range probs {
		scored[i] = scoredIdx{idx: i, score: p}
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	topN := 5
	if topN > len(scored) {
		topN = len(scored)
	}

	preds := make([]Prediction, topN)
	for i := 0; i < topN; i++ {
		idx := scored[i].idx
		lbl := fmt.Sprintf("Class %d", idx)
		if idx >= 0 && idx < len(vp.Labels) && vp.Labels[idx] != "" {
			lbl = vp.Labels[idx]
		}
		preds[i] = Prediction{
			Index:      idx,
			Label:      lbl,
			Confidence: scored[i].score,
		}
	}

	summary := formatVisionSummary(preds)

	res := &VisionResult{
		ModelPath:   vp.ModelPath,
		ImageWidth:  origW,
		ImageHeight: origH,
		ModelWidth:  vp.Width,
		ModelHeight: vp.Height,
		TopLabel:    preds[0].Label,
		TopScore:    preds[0].Confidence,
		Predictions: preds,
		Summary:     summary,
	}
	return res, nil
}

func formatVisionSummary(preds []Prediction) string {
	if len(preds) == 0 {
		return "No clear classification or objects detected."
	}

	top := preds[0]
	topPercent := top.Confidence * 100.0

	var sb strings.Builder
	if topPercent >= 20.0 {
		fmt.Fprintf(&sb, "The image appears to feature **%s** (%.1f%% confidence).", top.Label, topPercent)
	} else {
		fmt.Fprintf(&sb, "The image most closely resembles **%s** (%.1f%% confidence).", top.Label, topPercent)
	}

	if len(preds) > 1 {
		sb.WriteString("\nOther possibilities:")
		for i := 1; i < len(preds); i++ {
			p := preds[i]
			fmt.Fprintf(&sb, "\n- %s: %.1f%%", p.Label, p.Confidence*100.0)
		}
	}

	return sb.String()
}

func normalizeProbabilities(raw []float32) []float32 {
	if len(raw) == 0 {
		return raw
	}

	var sum float64
	hasNegative := false
	for _, v := range raw {
		if v < 0 {
			hasNegative = true
		}
		sum += float64(v)
	}

	if !hasNegative && sum > 0.90 && sum < 1.10 {
		probs := make([]float32, len(raw))
		copy(probs, raw)
		return probs
	}

	maxVal := float32(-math.MaxFloat32)
	for _, v := range raw {
		if v > maxVal {
			maxVal = v
		}
	}

	exps := make([]float64, len(raw))
	var expSum float64
	for i, v := range raw {
		e := math.Exp(float64(v - maxVal))
		exps[i] = e
		expSum += e
	}

	probs := make([]float32, len(raw))
	if expSum > 0 {
		for i, e := range exps {
			probs[i] = float32(e / expSum)
		}
	}
	return probs
}

// preprocessImage resizes an image to targetW x targetH and extracts RGB channels.
func preprocessImage(img image.Image, targetW, targetH, channels int, isNHWC bool) []byte {
	bounds := img.Bounds()
	srcW := bounds.Dx()
	srcH := bounds.Dy()

	outSize := targetW * targetH * channels
	buf := make([]byte, outSize)

	for y := 0; y < targetH; y++ {
		srcY := bounds.Min.Y + (y*srcH)/targetH
		for x := 0; x < targetW; x++ {
			srcX := bounds.Min.X + (x*srcW)/targetW
			c := img.At(srcX, srcY)
			r16, g16, b16, _ := c.RGBA()
			r := uint8(r16 >> 8)
			g := uint8(g16 >> 8)
			b := uint8(b16 >> 8)

			if isNHWC {
				idx := (y*targetW + x) * channels
				buf[idx] = r
				if channels > 1 {
					buf[idx+1] = g
				}
				if channels > 2 {
					buf[idx+2] = b
				}
			} else {
				spatialIdx := y*targetW + x
				buf[spatialIdx] = r
				if channels > 1 {
					buf[targetW*targetH+spatialIdx] = g
				}
				if channels > 2 {
					buf[2*targetW*targetH+spatialIdx] = b
				}
			}
		}
	}
	return buf
}
