package rknn

import (
	"fmt"
	"path/filepath"
	"runtime"
	"unsafe"

	"codeberg.org/indievisible/scooter/pkg/npu"
	// "github.com/ebitengine/purego"
)

type RKNNContext struct {
	handle Context
}

func (c *RKNNContext) Init(modelPath string) error {
	if err := LoadLibrary(); err != nil {
		return err
	}
	if !filepath.IsAbs(modelPath) {
		abs, err := filepath.Abs(modelPath)
		if err != nil {
			return err
		}
		modelPath = abs
	}

	modelPtr, modelSlice := stringToBytePtr(modelPath)
	defer runtime.KeepAlive(modelSlice)

	npu.Lock()
	defer npu.Unlock()

	var handle Context
	// Pass size=0, flag=0, extend=nil to load directly from file path
	if ret := rknnInit(&handle, unsafe.Pointer(modelPtr), 0, 0, nil); ret != 0 {
		return fmt.Errorf("rknn_init(%s) failed: %d", modelPath, ret)
	}
	c.handle = handle
	return nil
}

func (c *RKNNContext) SetCoreMask(mask uint32) error {
	npu.Lock()
	defer npu.Unlock()
	if ret := rknnSetCoreMask(c.handle, mask); ret != 0 {
		return fmt.Errorf("rknn_set_core_mask failed: %d", ret)
	}
	return nil
}

func (c *RKNNContext) QueryIO() (inputs []TensorAttr, outputs []TensorAttr, err error) {
	npu.Lock()
	defer npu.Unlock()

	var ioNum InputOutputNum
	if ret := rknnQuery(c.handle, RKNN_QUERY_IN_OUT_NUM, unsafe.Pointer(&ioNum), uint32(unsafe.Sizeof(ioNum))); ret != 0 {
		return nil, nil, fmt.Errorf("rknn_query IN_OUT_NUM failed: %d", ret)
	}

	inputs = make([]TensorAttr, ioNum.NInput)
	for i := uint32(0); i < ioNum.NInput; i++ {
		inputs[i].Index = i
		if ret := rknnQuery(c.handle, RKNN_QUERY_INPUT_ATTR, unsafe.Pointer(&inputs[i]), uint32(unsafe.Sizeof(inputs[i]))); ret != 0 {
			return nil, nil, fmt.Errorf("rknn_query INPUT_ATTR[%d] failed: %d", i, ret)
		}
	}

	outputs = make([]TensorAttr, ioNum.NOutput)
	for i := uint32(0); i < ioNum.NOutput; i++ {
		outputs[i].Index = i
		if ret := rknnQuery(c.handle, RKNN_QUERY_OUTPUT_ATTR, unsafe.Pointer(&outputs[i]), uint32(unsafe.Sizeof(outputs[i]))); ret != 0 {
			return nil, nil, fmt.Errorf("rknn_query OUTPUT_ATTR[%d] failed: %d", i, ret)
		}
	}
	return inputs, outputs, nil
}

// RunFloat32 executes a forward pass and copies all outputs as []float32 slices
// before releasing the C runtime buffers.
func (c *RKNNContext) RunFloat32(inputs []Input, nOutputs int) ([][]float32, error) {
	indices := make([]int, nOutputs)
	for i := 0; i < nOutputs; i++ {
		indices[i] = i
	}
	return c.RunFloat32Indices(inputs, indices)
}

// RunFloat32Indices executes a forward pass and queries specific output tensor indices.
func (c *RKNNContext) RunFloat32Indices(inputs []Input, outputIndices []int) ([][]float32, error) {
	npu.Lock()
	defer npu.Unlock()

	if len(inputs) > 0 {
		if ret := rknnInputsSet(c.handle, uint32(len(inputs)), &inputs[0]); ret != 0 {
			return nil, fmt.Errorf("rknn_inputs_set failed: %d", ret)
		}
	}

	if ret := rknnRun(c.handle, nil); ret != 0 {
		return nil, fmt.Errorf("rknn_run failed: %d", ret)
	}

	_, outputAttrs, err := c.queryIOUnlocked()
	if err != nil {
		return nil, err
	}

	results := make([][]float32, len(outputIndices))
	for i, idx := range outputIndices {
		if idx < 0 || idx >= len(outputAttrs) {
			return nil, fmt.Errorf("output index %d out of range (0-%d)", idx, len(outputAttrs)-1)
		}
		attr := outputAttrs[idx]
		nElems := int(attr.NElems)
		if nElems == 0 {
			nElems = int(attr.Size) / 4
			if nElems == 0 {
				nElems = 77 * 1280
			}
		}
		out := &Output{
			WantFloat:  1,
			IsPrealloc: 0,
			Index:      uint32(idx),
		}

		if ret := callRKNNOutputsGet(c.handle, 1, out); ret != 0 {
			return nil, fmt.Errorf("rknn_outputs_get failed for index %d: %d", idx, ret)
		}
		if out.Buf == nil || out.Size == 0 {
			callRKNNOutputsRelease(c.handle, 1, out)
			return nil, fmt.Errorf("rknn_outputs_get returned empty buffer for index %d", idx)
		}
		count := int(out.Size) / 4
		src := unsafe.Slice((*float32)(out.Buf), count)
		goSlice := make([]float32, count)
		copy(goSlice, src)
		callRKNNOutputsRelease(c.handle, 1, out)
		results[i] = goSlice
	}
	return results, nil
}

func (c *RKNNContext) queryIOUnlocked() (inputs []TensorAttr, outputs []TensorAttr, err error) {
	var ioNum InputOutputNum
	if ret := rknnQuery(c.handle, RKNN_QUERY_IN_OUT_NUM, unsafe.Pointer(&ioNum), uint32(unsafe.Sizeof(ioNum))); ret != 0 {
		return nil, nil, fmt.Errorf("rknn_query IN_OUT_NUM failed: %d", ret)
	}

	inputs = make([]TensorAttr, ioNum.NInput)
	for i := uint32(0); i < ioNum.NInput; i++ {
		inputs[i].Index = i
		if ret := rknnQuery(c.handle, RKNN_QUERY_INPUT_ATTR, unsafe.Pointer(&inputs[i]), uint32(unsafe.Sizeof(inputs[i]))); ret != 0 {
			return nil, nil, fmt.Errorf("rknn_query INPUT_ATTR[%d] failed: %d", i, ret)
		}
	}

	outputs = make([]TensorAttr, ioNum.NOutput)
	for i := uint32(0); i < ioNum.NOutput; i++ {
		outputs[i].Index = i
		if ret := rknnQuery(c.handle, RKNN_QUERY_OUTPUT_ATTR, unsafe.Pointer(&outputs[i]), uint32(unsafe.Sizeof(outputs[i]))); ret != 0 {
			return nil, nil, fmt.Errorf("rknn_query OUTPUT_ATTR[%d] failed: %d", i, ret)
		}
	}
	return inputs, outputs, nil
}

func (c *RKNNContext) CreateMem(size uint32) (*TensorMem, error) {
	npu.Lock()
	defer npu.Unlock()
	mem := rknnCreateMem(c.handle, size)
	if mem == nil {
		return nil, fmt.Errorf("rknn_create_mem failed for size %d", size)
	}
	return mem, nil
}

func (c *RKNNContext) DestroyMem(mem *TensorMem) error {
	if mem == nil {
		return nil
	}
	npu.Lock()
	defer npu.Unlock()
	if ret := rknnDestroyMem(c.handle, mem); ret != 0 {
		return fmt.Errorf("rknn_destroy_mem failed: %d", ret)
	}
	return nil
}

func (c *RKNNContext) SetIOMem(mem *TensorMem, attr *TensorAttr) error {
	npu.Lock()
	defer npu.Unlock()
	if ret := rknnSetIOMem(c.handle, mem, attr); ret != 0 {
		return fmt.Errorf("rknn_set_io_mem failed for tensor index %d: %d", attr.Index, ret)
	}
	return nil
}

func (c *RKNNContext) Close() error {
	npu.Lock()
	defer npu.Unlock()
	if c.handle == 0 {
		return nil
	}
	ret := rknnDestroy(c.handle)
	c.handle = 0
	if ret != 0 {
		return fmt.Errorf("rknn_destroy failed: %d", ret)
	}
	return nil
}

func stringToBytePtr(s string) (*byte, []byte) {
	b := append([]byte(s), 0)
	return &b[0], b
}

/*
//go:noinline
func callRKNNOutputsGet(handle Context, nOutputs uint32, out *Output) int32 {
	r1, _, _ := purego.SyscallN(
		rknnOutputsGetSym,
		uintptr(handle),
		uintptr(nOutputs),
		uintptr(unsafe.Pointer(out)),
		0,
	)
	runtime.KeepAlive(out)
	return int32(r1)
}

//go:noinline
func callRKNNOutputsRelease(handle Context, nOutputs uint32, out *Output) int32 {
	r1, _, _ := purego.SyscallN(
		rknnOutputsReleaseSym,
		uintptr(handle),
		uintptr(nOutputs),
		uintptr(unsafe.Pointer(out)),
	)
	runtime.KeepAlive(out)
	return int32(r1)
}
*/
