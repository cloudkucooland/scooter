package rknn

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"unsafe"

	"codeberg.org/indievisible/scooter/pkg/npu"
)

const (
	sdxlVAEScaleFactor float32 = 0.13025
	sd15VAEScaleFactor float32 = 0.18215
)

type Pipeline struct {
	ModelDir      string
	Tok           *Tokenizer
	Sched         *Scheduler
	GuidanceScale float32

	IsSD15       bool
	UseCFG       bool
	VAEScale     float32
	Width        int
	Height       int
	UNetInCount  int
	UNetInIsNHWC bool
	VAEInIsNHWC  bool
	VAEIsNHWC    bool
	UNetOutSize  int
	VAEOutSize   int
}

// LoadPipeline validates the RKNN model directory and probes architecture,
// scheduler mode, tensor dimensions, and layouts without keeping all submodels
// resident in NPU/CMA memory simultaneously.
func LoadPipeline(modelDir string, tok *Tokenizer, steps int) (*Pipeline, error) {
	p := &Pipeline{
		ModelDir:      modelDir,
		Tok:           tok,
		GuidanceScale: 1.5,
		VAEScale:      sdxlVAEScaleFactor,
	}

	// Detect SD 1.5 vs SDXL by checking for text_encoder_2/model.rknn
	if _, err := os.Stat(filepath.Join(modelDir, "text_encoder_2", "model.rknn")); os.IsNotExist(err) {
		p.IsSD15 = true
		p.VAEScale = sd15VAEScaleFactor
	}

	// Probe UNet input resolution, layout, and input/output counts
	unetCtx, err := p.openSubmodel("unet")
	if err != nil {
		return nil, err
	}
	unetIn, unetOut, err := unetCtx.QueryIO()
	_ = unetCtx.Close()
	if err != nil {
		return nil, err
	}
	p.UNetInCount = len(unetIn)
	p.UNetOutSize = len(unetOut)

	if len(unetIn) > 0 && unetIn[0].NDims == 4 {
		if unetIn[0].Fmt == RKNN_TENSOR_NHWC {
			p.UNetInIsNHWC = true
			p.Height = int(unetIn[0].Dims[1]) * 8
			p.Width = int(unetIn[0].Dims[2]) * 8
		} else {
			p.Height = int(unetIn[0].Dims[2]) * 8
			p.Width = int(unetIn[0].Dims[3]) * 8
		}
	}

	// Select scheduler mode:
	// - 3-input SD 1.5 UNet (sample, timestep, encoder_hidden_states) -> standard Euler with CFG
	// - 4-input SD 1.5 UNet (+ timestep_cond) or 6-input SDXL UNet -> LCM
	schedMode := "lcm"
	if p.IsSD15 && p.UNetInCount == 3 {
		schedMode = "euler"
		p.UseCFG = true
		p.GuidanceScale = 7.5
	}
	p.Sched = NewScheduler(steps, schedMode)

	// Probe VAE input and output layouts (NHWC vs NCHW)
	vaeCtx, err := p.openSubmodel("vae_decoder")
	if err != nil {
		return nil, err
	}
	vaeIn, vaeOut, err := vaeCtx.QueryIO()
	_ = vaeCtx.Close()
	if err != nil {
		return nil, err
	}
	if len(vaeIn) > 0 && vaeIn[0].Fmt == RKNN_TENSOR_NHWC {
		p.VAEInIsNHWC = true
	}
	p.VAEOutSize = len(vaeOut)
	if len(vaeOut) > 0 && vaeOut[0].Fmt == RKNN_TENSOR_NHWC {
		p.VAEIsNHWC = true
	}

	return p, nil
}

func (p *Pipeline) openSubmodel(sub string) (*RKNNContext, error) {
	ctx := &RKNNContext{}
	path := filepath.Join(p.ModelDir, sub, "model.rknn")
	if err := ctx.Init(path); err != nil {
		return nil, err
	}
	if err := ctx.SetCoreMask(RKNN_NPU_CORE_0_1_2); err != nil {
		_ = ctx.Close()
		return nil, err
	}
	return ctx, nil
}

func (p *Pipeline) Close() {}

func (p *Pipeline) Generate(prompt string, seed int64) (*image.RGBA, error) {
	return p.GenerateWithNegative(context.Background(), prompt, "", seed, nil)
}

func (p *Pipeline) GenerateWithContext(ctx context.Context, prompt string, seed int64, progressFn func(step, total int)) (*image.RGBA, error) {
	return p.GenerateWithNegative(ctx, prompt, "", seed, progressFn)
}

func (p *Pipeline) GenerateWithNegative(ctx context.Context, prompt, negativePrompt string, seed int64, progressFn func(step, total int)) (*image.RGBA, error) {
	if p.IsSD15 {
		ids, _, err := p.Tok.EncodeDual(prompt)
		if err != nil {
			return nil, fmt.Errorf("tokenization failed: %w", err)
		}
		condEmbeds, err := p.encodeSingleText(ids)
		if err != nil {
			return nil, fmt.Errorf("text encoding failed: %w", err)
		}
		var uncondEmbeds []float32
		if p.UseCFG {
			uncondIDs, _, err := p.Tok.EncodeDual(negativePrompt)
			if err != nil {
				return nil, fmt.Errorf("uncond tokenization failed: %w", err)
			}
			uncondEmbeds, err = p.encodeSingleText(uncondIDs)
			if err != nil {
				return nil, fmt.Errorf("uncond text encoding failed: %w", err)
			}
		}
		latents, err := p.runDiffusionSD15(ctx, condEmbeds, uncondEmbeds, seed, progressFn)
		if err != nil {
			return nil, fmt.Errorf("diffusion loop failed: %w", err)
		}
		img, err := p.decodeVAE(latents)
		if err != nil {
			return nil, fmt.Errorf("VAE decode failed: %w", err)
		}
		return img, nil
	}

	ids1, ids2, err := p.Tok.EncodeDual(prompt)
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %w", err)
	}

	promptEmbeds, pooledEmbeds, err := p.encodeDualText(ids1, ids2)
	if err != nil {
		return nil, fmt.Errorf("text encoding failed: %w", err)
	}

	latents, err := p.runDiffusionSDXL(ctx, promptEmbeds, pooledEmbeds, seed, progressFn)
	if err != nil {
		return nil, fmt.Errorf("diffusion loop failed: %w", err)
	}

	img, err := p.decodeVAE(latents)
	if err != nil {
		return nil, fmt.Errorf("VAE decode failed: %w", err)
	}
	return img, nil
}

func runSubmodelZeroCopy(ctx *RKNNContext, writeInput func(mem *TensorMem, attr TensorAttr)) ([][]float32, error) {
	inputsAttr, outputsAttr, err := ctx.QueryIO()
	if err != nil {
		return nil, err
	}
	if len(inputsAttr) == 0 {
		return nil, fmt.Errorf("submodel has no inputs")
	}

	inMem, err := ctx.CreateMem(inputsAttr[0].Size)
	if err != nil {
		return nil, fmt.Errorf("create input mem: %w", err)
	}
	defer func() { _ = ctx.DestroyMem(inMem) }()

	if err := ctx.SetIOMem(inMem, &inputsAttr[0]); err != nil {
		return nil, fmt.Errorf("set input io mem: %w", err)
	}

	outMems := make([]*TensorMem, len(outputsAttr))
	for i, attr := range outputsAttr {
		mem, err := ctx.CreateMem(attr.Size)
		if err != nil {
			for j := 0; j < i; j++ {
				_ = ctx.DestroyMem(outMems[j])
			}
			return nil, fmt.Errorf("create output mem %d: %w", i, err)
		}
		outMems[i] = mem
		if err := ctx.SetIOMem(mem, &attr); err != nil {
			for j := 0; j <= i; j++ {
				_ = ctx.DestroyMem(outMems[j])
			}
			return nil, fmt.Errorf("set output io mem %d: %w", i, err)
		}
	}
	defer func() {
		for _, mem := range outMems {
			_ = ctx.DestroyMem(mem)
		}
	}()

	writeInput(inMem, inputsAttr[0])

	npu.Lock()
	ret := rknnRun(ctx.handle, nil)
	npu.Unlock()
	if ret != 0 {
		return nil, fmt.Errorf("rknn_run failed: %d", ret)
	}

	results := make([][]float32, len(outputsAttr))
	for i, attr := range outputsAttr {
		if attr.Type == RKNN_TENSOR_FLOAT16 {
			nElems := int(attr.Size) / 2
			u16Slice := unsafe.Slice((*uint16)(outMems[i].VirtAddr), nElems)
			out := make([]float32, nElems)
			for idx, u := range u16Slice {
				out[idx] = float16ToFloat32(u)
			}
			results[i] = out
		} else {
			nElems := int(attr.Size) / 4
			f32Slice := unsafe.Slice((*float32)(outMems[i].VirtAddr), nElems)
			out := make([]float32, nElems)
			copy(out, f32Slice)
			results[i] = out
		}
	}
	return results, nil
}

func (p *Pipeline) encodeSingleText(ids []int32) ([]float32, error) {
	enc, err := p.openSubmodel("text_encoder")
	if err != nil {
		return nil, err
	}
	out, err := runSubmodelZeroCopy(enc, func(mem *TensorMem, attr TensorAttr) {
		if attr.Type == RKNN_TENSOR_INT64 {
			ids64 := make([]int64, len(ids))
			for i, v := range ids {
				ids64[i] = int64(v)
			}
			copyInt64ToMem(mem, ids64)
		} else {
			dst := unsafe.Slice((*int32)(mem.VirtAddr), len(ids))
			copy(dst, ids)
		}
	})
	_ = enc.Close()
	if err != nil {
		return nil, err
	}
	if len(out) < 1 {
		return nil, fmt.Errorf("text_encoder returned no outputs")
	}
	return out[0], nil
}

func (p *Pipeline) encodeDualText(ids1, ids2 []int32) (promptEmbeds []float32, pooledEmbeds []float32, err error) {
	hidden1, err := p.encodeSingleText(ids1)
	if err != nil {
		return nil, nil, err
	}

	enc2, err := p.openSubmodel("text_encoder_2")
	if err != nil {
		return nil, nil, err
	}
	out2, err := runSubmodelZeroCopy(enc2, func(mem *TensorMem, attr TensorAttr) {
		if attr.Type == RKNN_TENSOR_INT32 {
			dst := unsafe.Slice((*int32)(mem.VirtAddr), len(ids2))
			copy(dst, ids2)
		} else {
			ids64 := make([]int64, len(ids2))
			for i, v := range ids2 {
				ids64[i] = int64(v)
			}
			copyInt64ToMem(mem, ids64)
		}
	})
	_ = enc2.Close()
	if err != nil {
		return nil, nil, err
	}
	if len(out2) < 2 {
		return nil, nil, fmt.Errorf("text_encoder_2 returned %d outputs, expected 2", len(out2))
	}
	pooledEmbeds = out2[0]
	hidden2 := out2[1]

	const seqLen = 77
	promptEmbeds = make([]float32, seqLen*(768+1280))
	for t := 0; t < seqLen; t++ {
		copy(promptEmbeds[t*2048:t*2048+768], hidden1[t*768:(t+1)*768])
		copy(promptEmbeds[t*2048+768:(t+1)*2048], hidden2[t*1280:(t+1)*1280])
	}
	return promptEmbeds, pooledEmbeds, nil
}

func (p *Pipeline) runDiffusionSD15(ctx context.Context, condEmbeds, uncondEmbeds []float32, seed int64, progressFn func(step, total int)) ([]float32, error) {
	unet, err := p.openSubmodel("unet")
	if err != nil {
		return nil, err
	}
	defer func() { _ = unet.Close() }()

	inputsAttr, outputsAttr, err := unet.QueryIO()
	if err != nil {
		return nil, err
	}

	p.Sched.Reset(seed)
	rng := rand.New(rand.NewSource(seed))
	latentH := p.Height / 8
	latentW := p.Width / 8
	latents := make([]float32, 4*latentH*latentW)

	initSigma := float32(p.Sched.InitNoiseSigma())
	for i := range latents {
		latents[i] = float32(rng.NormFloat64()) * initSigma
	}

	guidanceScale := p.GuidanceScale
	if guidanceScale == 0 {
		if p.UseCFG {
			guidanceScale = 7.5
		} else {
			guidanceScale = 1.5
		}
	}
	timestepCond := getGuidanceScaleEmbedding(guidanceScale)

	inputMems := make([]*TensorMem, len(inputsAttr))
	for i, attr := range inputsAttr {
		size := attr.Size
		if i == 0 && uint32(len(latents)*4) > size {
			size = uint32(len(latents) * 4)
		} else if i == 2 && uint32(len(condEmbeds)*4) > size {
			size = uint32(len(condEmbeds) * 4)
		} else if i == 3 && uint32(len(timestepCond)*4) > size {
			size = uint32(len(timestepCond) * 4)
		}
		mem, err := unet.CreateMem(size)
		if err != nil {
			for j := 0; j < i; j++ {
				_ = unet.DestroyMem(inputMems[j])
			}
			return nil, fmt.Errorf("create input mem %d: %w", i, err)
		}
		inputMems[i] = mem
		if err := unet.SetIOMem(mem, &attr); err != nil {
			for j := 0; j <= i; j++ {
				_ = unet.DestroyMem(inputMems[j])
			}
			return nil, fmt.Errorf("set io mem for input %d: %w", i, err)
		}
	}
	defer func() {
		for _, mem := range inputMems {
			_ = unet.DestroyMem(mem)
		}
	}()

	outputMems := make([]*TensorMem, len(outputsAttr))
	for i, attr := range outputsAttr {
		mem, err := unet.CreateMem(attr.Size)
		if err != nil {
			for j := 0; j < i; j++ {
				_ = unet.DestroyMem(outputMems[j])
			}
			return nil, fmt.Errorf("create output mem %d: %w", i, err)
		}
		outputMems[i] = mem
		if err := unet.SetIOMem(mem, &attr); err != nil {
			for j := 0; j <= i; j++ {
				_ = unet.DestroyMem(outputMems[j])
			}
			return nil, fmt.Errorf("set io mem for output %d: %w", i, err)
		}
	}
	defer func() {
		for _, mem := range outputMems {
			_ = unet.DestroyMem(mem)
		}
	}()

	if len(inputsAttr) > 3 {
		copyTensorFloat32ToMem(inputMems[3], inputsAttr[3], timestepCond)
	}
	if !p.UseCFG && len(inputsAttr) > 2 {
		copyTensorFloat32ToMem(inputMems[2], inputsAttr[2], condEmbeds)
	}

	readUNetOut := func() []float32 {
		outSize := outputsAttr[0].Size
		if outputsAttr[0].Type == RKNN_TENSOR_FLOAT16 {
			nElems := int(outSize) / 2
			u16Slice := unsafe.Slice((*uint16)(outputMems[0].VirtAddr), nElems)
			out := make([]float32, nElems)
			for idx, u := range u16Slice {
				out[idx] = float16ToFloat32(u)
			}
			return out
		}
		nElems := int(outSize) / 4
		cSlice := unsafe.Slice((*float32)(outputMems[0].VirtAddr), nElems)
		out := make([]float32, nElems)
		copy(out, cSlice)
		return out
	}

	writeTimestep := func(t int) {
		if inputsAttr[1].Type == RKNN_TENSOR_INT64 {
			copyInt64ToMem(inputMems[1], []int64{int64(t)})
		} else if inputsAttr[1].Type == RKNN_TENSOR_INT32 {
			dst := unsafe.Slice((*int32)(inputMems[1].VirtAddr), 1)
			dst[0] = int32(t)
		} else {
			copyTensorFloat32ToMem(inputMems[1], inputsAttr[1], []float32{float32(t)})
		}
	}

	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		timestep, ok := p.Sched.NextTimestep()
		if !ok {
			break
		}
		if progressFn != nil {
			progressFn(p.Sched.stepIndex+1, p.Sched.Steps)
		}
		scaledLatents := p.Sched.ScaleModelInput(latents, timestep)
		if p.UNetInIsNHWC {
			scaledLatents = nchwToNHWC4(scaledLatents, latentH, latentW)
		}

		copyTensorFloat32ToMem(inputMems[0], inputsAttr[0], scaledLatents)
		writeTimestep(timestep)

		var noisePred []float32
		if p.UseCFG {
			// Unconditional pass
			copyTensorFloat32ToMem(inputMems[2], inputsAttr[2], uncondEmbeds)
			npu.Lock()
			ret := rknnRun(unet.handle, nil)
			npu.Unlock()
			if ret != 0 {
				return nil, fmt.Errorf("rknn_run (uncond) failed: %d", ret)
			}
			noiseUncond := readUNetOut()

			// Conditional pass
			copyTensorFloat32ToMem(inputMems[2], inputsAttr[2], condEmbeds)
			npu.Lock()
			ret = rknnRun(unet.handle, nil)
			npu.Unlock()
			if ret != 0 {
				return nil, fmt.Errorf("rknn_run (cond) failed: %d", ret)
			}
			noiseCond := readUNetOut()

			noisePred = make([]float32, len(noiseCond))
			for i := range noisePred {
				noisePred[i] = noiseUncond[i] + guidanceScale*(noiseCond[i]-noiseUncond[i])
			}
		} else {
			npu.Lock()
			ret := rknnRun(unet.handle, nil)
			npu.Unlock()
			if ret != 0 {
				return nil, fmt.Errorf("rknn_run failed: %d", ret)
			}
			noisePred = readUNetOut()
		}

		latents = p.Sched.Step(noisePred, timestep, latents)
	}
	return latents, nil
}

func (p *Pipeline) runDiffusionSDXL(ctx context.Context, promptEmbeds, pooledEmbeds []float32, seed int64, progressFn func(step, total int)) ([]float32, error) {
	unet, err := p.openSubmodel("unet")
	if err != nil {
		return nil, err
	}
	defer func() { _ = unet.Close() }()

	inputsAttr, outputsAttr, err := unet.QueryIO()
	if err != nil {
		return nil, err
	}

	p.Sched.Reset(seed)
	rng := rand.New(rand.NewSource(seed))
	latentH := p.Height / 8
	latentW := p.Width / 8
	latents := make([]float32, 4*latentH*latentW)

	initSigma := float32(p.Sched.InitNoiseSigma())
	for i := range latents {
		latents[i] = float32(rng.NormFloat64()) * initSigma
	}

	guidanceScale := p.GuidanceScale
	if guidanceScale == 0 {
		guidanceScale = 1.5
	}
	timestepCond := getGuidanceScaleEmbedding(guidanceScale)

	timeIDs := []float32{
		float32(p.Height), float32(p.Width),
		0, 0,
		float32(p.Height), float32(p.Width),
	}

	// Allocate zero-copy input and output tensor memories
	inputMems := make([]*TensorMem, len(inputsAttr))
	for i, attr := range inputsAttr {
		size := attr.Size
		if i == 0 {
			reqSize := uint32(len(latents) * 4)
			if size < reqSize {
				size = reqSize
			}
		} else if i == 2 && uint32(len(promptEmbeds)*4) > size {
			size = uint32(len(promptEmbeds) * 4)
		} else if i == 3 && uint32(len(timestepCond)*4) > size {
			size = uint32(len(timestepCond) * 4)
		} else if i == 4 && uint32(len(timeIDs)*4) > size {
			size = uint32(len(timeIDs) * 4)
		} else if i == 5 && uint32(len(pooledEmbeds)*4) > size {
			size = uint32(len(pooledEmbeds) * 4)
		}
		mem, err := unet.CreateMem(size)
		if err != nil {
			for j := 0; j < i; j++ {
				_ = unet.DestroyMem(inputMems[j])
			}
			return nil, fmt.Errorf("create input mem %d: %w", i, err)
		}
		inputMems[i] = mem
		if err := unet.SetIOMem(mem, &attr); err != nil {
			for j := 0; j <= i; j++ {
				_ = unet.DestroyMem(inputMems[j])
			}
			return nil, fmt.Errorf("set io mem for input %d: %w", i, err)
		}
	}
	defer func() {
		for _, mem := range inputMems {
			_ = unet.DestroyMem(mem)
		}
	}()

	outputMems := make([]*TensorMem, len(outputsAttr))
	for i, attr := range outputsAttr {
		mem, err := unet.CreateMem(attr.Size)
		if err != nil {
			for j := 0; j < i; j++ {
				_ = unet.DestroyMem(outputMems[j])
			}
			return nil, fmt.Errorf("create output mem %d: %w", i, err)
		}
		outputMems[i] = mem
		if err := unet.SetIOMem(mem, &attr); err != nil {
			for j := 0; j <= i; j++ {
				_ = unet.DestroyMem(outputMems[j])
			}
			return nil, fmt.Errorf("set io mem for output %d: %w", i, err)
		}
	}
	defer func() {
		for _, mem := range outputMems {
			_ = unet.DestroyMem(mem)
		}
	}()

	// Copy invariant inputs once
	if len(inputsAttr) > 2 {
		copyTensorFloat32ToMem(inputMems[2], inputsAttr[2], promptEmbeds)
	}
	if len(inputsAttr) > 3 {
		copyTensorFloat32ToMem(inputMems[3], inputsAttr[3], timestepCond)
	}
	if len(inputsAttr) > 4 {
		copyTensorFloat32ToMem(inputMems[4], inputsAttr[4], timeIDs)
	}
	if len(inputsAttr) > 5 {
		copyTensorFloat32ToMem(inputMems[5], inputsAttr[5], pooledEmbeds)
	}

	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		timestep, ok := p.Sched.NextTimestep()
		if !ok {
			break
		}
		if progressFn != nil {
			progressFn(p.Sched.stepIndex+1, p.Sched.Steps)
		}
		scaledLatents := p.Sched.ScaleModelInput(latents, timestep)
		if p.UNetInIsNHWC {
			scaledLatents = nchwToNHWC4(scaledLatents, latentH, latentW)
		}

		copyTensorFloat32ToMem(inputMems[0], inputsAttr[0], scaledLatents)

		tScalar := []int64{int64(timestep)}
		copyInt64ToMem(inputMems[1], tScalar)

		npu.Lock()
		ret := rknnRun(unet.handle, nil)
		npu.Unlock()
		if ret != 0 {
			return nil, fmt.Errorf("rknn_run failed: %d", ret)
		}

		outSize := outputsAttr[0].Size
		outType := outputsAttr[0].Type
		var outData []float32
		if outType == RKNN_TENSOR_FLOAT16 {
			nElems := int(outSize) / 2
			u16Slice := unsafe.Slice((*uint16)(outputMems[0].VirtAddr), nElems)
			outData = make([]float32, nElems)
			for idx, u := range u16Slice {
				outData[idx] = float16ToFloat32(u)
			}
		} else {
			nElems := int(outSize) / 4
			cSlice := unsafe.Slice((*float32)(outputMems[0].VirtAddr), nElems)
			outData = make([]float32, nElems)
			copy(outData, cSlice)
		}

		latents = p.Sched.Step(outData, timestep, latents)
	}
	return latents, nil
}

func copyInt64ToMem(mem *TensorMem, src []int64) {
	if len(src) == 0 || mem.VirtAddr == nil {
		return
	}
	dest := unsafe.Slice((*int64)(mem.VirtAddr), len(src))
	copy(dest, src)
}

func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 0x01
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff

	var fbits uint32
	if exp == 0 {
		if mant == 0 {
			fbits = sign << 31
		} else {
			exp = 127 - 15 - 1
			for (mant & 0x400) == 0 {
				mant <<= 1
				exp--
			}
			mant &= 0x3ff
			fbits = (sign << 31) | (exp << 23) | (mant << 13)
		}
	} else if exp == 31 {
		fbits = (sign << 31) | (255 << 23) | (mant << 13)
	} else {
		fbits = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13)
	}
	return *(*float32)(unsafe.Pointer(&fbits))
}

func (p *Pipeline) decodeVAE(latents []float32) (*image.RGBA, error) {
	vae, err := p.openSubmodel("vae_decoder")
	if err != nil {
		return nil, err
	}
	defer func() { _ = vae.Close() }()

	inputsAttr, outputsAttr, err := vae.QueryIO()
	if err != nil {
		return nil, err
	}

	scaleFactor := p.VAEScale
	if scaleFactor == 0 {
		scaleFactor = sdxlVAEScaleFactor
	}

	scaled := make([]float32, len(latents))
	for i, v := range latents {
		scaled[i] = v / scaleFactor
	}

	if p.VAEInIsNHWC {
		scaled = nchwToNHWC4(scaled, p.Height/8, p.Width/8)
	}

	inputMems := make([]*TensorMem, len(inputsAttr))
	for i, attr := range inputsAttr {
		size := attr.Size
		if i == 0 {
			reqSize := uint32(len(scaled) * 4)
			if size < reqSize {
				size = reqSize
			}
		}
		mem, err := vae.CreateMem(size)
		if err != nil {
			for j := 0; j < i; j++ {
				_ = vae.DestroyMem(inputMems[j])
			}
			return nil, fmt.Errorf("vae create input mem %d: %w", i, err)
		}
		inputMems[i] = mem
		if err := vae.SetIOMem(mem, &attr); err != nil {
			for j := 0; j <= i; j++ {
				_ = vae.DestroyMem(inputMems[j])
			}
			return nil, fmt.Errorf("vae set io mem for input %d: %w", i, err)
		}
	}
	defer func() {
		for _, mem := range inputMems {
			_ = vae.DestroyMem(mem)
		}
	}()

	outputMems := make([]*TensorMem, len(outputsAttr))
	for i, attr := range outputsAttr {
		mem, err := vae.CreateMem(attr.Size)
		if err != nil {
			for j := 0; j < i; j++ {
				_ = vae.DestroyMem(outputMems[j])
			}
			return nil, fmt.Errorf("vae create output mem %d: %w", i, err)
		}
		outputMems[i] = mem
		if err := vae.SetIOMem(mem, &attr); err != nil {
			for j := 0; j <= i; j++ {
				_ = vae.DestroyMem(outputMems[j])
			}
			return nil, fmt.Errorf("vae set io mem for output %d: %w", i, err)
		}
	}
	defer func() {
		for _, mem := range outputMems {
			_ = vae.DestroyMem(mem)
		}
	}()

	if len(inputsAttr) > 0 {
		copyTensorFloat32ToMem(inputMems[0], inputsAttr[0], scaled)
	}
	runtime.KeepAlive(scaled)

	npu.Lock()
	ret := rknnRun(vae.handle, nil)
	npu.Unlock()
	if ret != 0 {
		return nil, fmt.Errorf("vae rknn_run failed: %d", ret)
	}

	outSize := outputsAttr[0].Size
	outType := outputsAttr[0].Type
	var pixels []float32
	if outType == RKNN_TENSOR_FLOAT16 {
		nElems := int(outSize) / 2
		u16Slice := unsafe.Slice((*uint16)(outputMems[0].VirtAddr), nElems)
		pixels = make([]float32, nElems)
		for idx, u := range u16Slice {
			pixels[idx] = float16ToFloat32(u)
		}
	} else {
		nElems := int(outSize) / 4
		cSlice := unsafe.Slice((*float32)(outputMems[0].VirtAddr), nElems)
		pixels = make([]float32, nElems)
		copy(pixels, cSlice)
	}

	img := image.NewRGBA(image.Rect(0, 0, p.Width, p.Height))
	planeSize := p.Width * p.Height

	toUint8 := func(v float32) uint8 {
		clamped := math.Max(0, math.Min(1, float64(v*0.5+0.5)))
		return uint8(math.Round(clamped * 255.0))
	}

	for y := 0; y < p.Height; y++ {
		for x := 0; x < p.Width; x++ {
			idx := y*p.Width + x
			var r, g, b float32
			if p.VAEIsNHWC {
				r = pixels[idx*3+0]
				g = pixels[idx*3+1]
				b = pixels[idx*3+2]
			} else {
				r = pixels[0*planeSize+idx]
				g = pixels[1*planeSize+idx]
				b = pixels[2*planeSize+idx]
			}
			img.SetRGBA(x, y, color.RGBA{R: toUint8(r), G: toUint8(g), B: toUint8(b), A: 255})
		}
	}
	return img, nil
}

func getGuidanceScaleEmbedding(guidanceScale float32) []float32 {
	const halfDim = 128
	w := (guidanceScale - 1.0) * 1000.0
	emb := make([]float32, 256)
	logStep := math.Log(10000.0) / float64(halfDim)
	for i := 0; i < halfDim; i++ {
		freq := math.Exp(-float64(i) * logStep)
		angle := float64(w) * freq
		emb[i] = float32(math.Cos(angle))
		emb[i+halfDim] = float32(math.Sin(angle))
	}
	return emb
}

func nchwToNHWC4(nchw []float32, h, w int) []float32 {
	nhwc := make([]float32, len(nchw))
	plane := h * w
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			hw := y*w + x
			dst := hw * 4
			nhwc[dst+0] = nchw[0*plane+hw]
			nhwc[dst+1] = nchw[1*plane+hw]
			nhwc[dst+2] = nchw[2*plane+hw]
			nhwc[dst+3] = nchw[3*plane+hw]
		}
	}
	return nhwc
}

func copyTensorFloat32ToMem(mem *TensorMem, attr TensorAttr, src []float32) {
	if len(src) == 0 || mem.VirtAddr == nil {
		return
	}
	if attr.Type == RKNN_TENSOR_FLOAT16 {
		dest := unsafe.Slice((*uint16)(mem.VirtAddr), len(src))
		for i, v := range src {
			dest[i] = float32ToFloat16(v)
		}
		return
	}
	dest := unsafe.Slice((*float32)(mem.VirtAddr), len(src))
	copy(dest, src)
}

func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x7fffff

	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant = (mant | 0x800000) >> uint(1-exp)
		return sign | uint16((mant+0x1000)>>13)
	}
	if exp >= 31 {
		if exp == 128 && mant != 0 {
			return sign | 0x7e00 | uint16(mant>>13)
		}
		return sign | 0x7c00
	}
	return sign | uint16(exp<<10) | uint16((mant+0x1000)>>13)
}
