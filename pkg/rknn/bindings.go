package rknn

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"unsafe"

	"codeberg.org/indievisible/scooter/pkg/npu"
	"github.com/ebitengine/purego"
)

func Lock()   { npu.Lock() }
func Unlock() { npu.Unlock() }

type Context uint64 // typedef uint64_t rknn_context

const (
	RKNN_MAX_DIMS     = 16
	RKNN_MAX_NAME_LEN = 256

	RKNN_NPU_CORE_AUTO  uint32 = 0
	RKNN_NPU_CORE_0     uint32 = 1
	RKNN_NPU_CORE_1     uint32 = 2
	RKNN_NPU_CORE_2     uint32 = 4
	RKNN_NPU_CORE_0_1   uint32 = 3
	RKNN_NPU_CORE_0_1_2 uint32 = 7

	RKNN_QUERY_IN_OUT_NUM  uint32 = 0
	RKNN_QUERY_INPUT_ATTR  uint32 = 1
	RKNN_QUERY_OUTPUT_ATTR uint32 = 2

	RKNN_TENSOR_FLOAT32 uint32 = 0
	RKNN_TENSOR_FLOAT16 uint32 = 1
	RKNN_TENSOR_INT8    uint32 = 2
	RKNN_TENSOR_UINT8   uint32 = 3
	RKNN_TENSOR_INT16   uint32 = 4
	RKNN_TENSOR_UINT16  uint32 = 5
	RKNN_TENSOR_INT32   uint32 = 6
	RKNN_TENSOR_UINT32  uint32 = 7
	RKNN_TENSOR_INT64   uint32 = 8
	RKNN_TENSOR_BOOL    uint32 = 9

	RKNN_TENSOR_NCHW      uint32 = 0
	RKNN_TENSOR_NHWC      uint32 = 1
	RKNN_TENSOR_NC1HWC2   uint32 = 2
	RKNN_TENSOR_UNDEFINED uint32 = 3
)

type InputOutputNum struct {
	NInput  uint32
	NOutput uint32
}

type TensorAttr struct {
	Index       uint32
	NDims       uint32
	Dims        [RKNN_MAX_DIMS]uint32
	Name        [RKNN_MAX_NAME_LEN]byte
	NElems      uint32
	Size        uint32
	Fmt         uint32
	Type        uint32
	QntType     uint32
	Fl          int8
	_           [3]byte
	Zp          int32
	Scale       float32
	WStride     uint32
	SizeWithPad uint32
	PassThrough uint8
	HStride     uint32
}

type Input struct {
	Index       uint32
	_           [4]byte
	Buf         unsafe.Pointer
	Size        uint32
	PassThrough uint8
	Type        uint32
	Fmt         uint32
}

type Output struct {
	WantFloat  uint8
	IsPrealloc uint8
	_          [2]byte
	Index      uint32
	Buf        unsafe.Pointer
	Size       uint32
	_          [4]byte
}

type TensorMem struct {
	VirtAddr unsafe.Pointer
	PhysAddr uint64
	Fd       int32
	Offset   int32
	Size     uint32
	Flag     uint32
	PrivData unsafe.Pointer
}

var (
	libOnce sync.Once
	libErr  error
	lib     uintptr

	rknnInit              func(ctx *Context, model unsafe.Pointer, size uint32, flag uint32, extend unsafe.Pointer) int32
	rknnDestroy           func(ctx Context) int32
	rknnSetCoreMask       func(ctx Context, coreMask uint32) int32
	rknnQuery             func(ctx Context, cmd uint32, info unsafe.Pointer, size uint32) int32
	rknnInputsSet         func(ctx Context, nInputs uint32, inputs *Input) int32
	rknnRun               func(ctx Context, extend unsafe.Pointer) int32
	rknnOutputsGetSym     uintptr
	rknnOutputsReleaseSym uintptr
	rknnCreateMem         func(ctx Context, size uint32) *TensorMem
	rknnDestroyMem        func(ctx Context, mem *TensorMem) int32
	rknnSetIOMem          func(ctx Context, mem *TensorMem, attr *TensorAttr) int32
)

func getRKNNLibPath() string {
	if p := os.Getenv("RKNN_LIB"); p != "" {
		return p
	}
	return "/usr/lib/librknnrt.so"
}

func LoadLibrary() error {
	libOnce.Do(func() {
		libPath := getRKNNLibPath()
		if !filepath.IsAbs(libPath) {
			libPath = filepath.Join(".", libPath)
		}
		lib, libErr = purego.Dlopen(libPath, purego.RTLD_NOW)
		if libErr != nil {
			libErr = fmt.Errorf("failed to dlopen %s: %w", libPath, libErr)
			return
		}
		purego.RegisterLibFunc(&rknnInit, lib, "rknn_init")
		purego.RegisterLibFunc(&rknnDestroy, lib, "rknn_destroy")
		purego.RegisterLibFunc(&rknnSetCoreMask, lib, "rknn_set_core_mask")
		purego.RegisterLibFunc(&rknnQuery, lib, "rknn_query")
		purego.RegisterLibFunc(&rknnInputsSet, lib, "rknn_inputs_set")
		purego.RegisterLibFunc(&rknnRun, lib, "rknn_run")
		rknnOutputsGetSym, libErr = purego.Dlsym(lib, "rknn_outputs_get")
		if libErr != nil {
			return
		}
		rknnOutputsReleaseSym, libErr = purego.Dlsym(lib, "rknn_outputs_release")
		if libErr != nil {
			return
		}
		purego.RegisterLibFunc(&rknnCreateMem, lib, "rknn_create_mem")
		purego.RegisterLibFunc(&rknnDestroyMem, lib, "rknn_destroy_mem")
		purego.RegisterLibFunc(&rknnSetIOMem, lib, "rknn_set_io_mem")
	})
	return libErr
}
