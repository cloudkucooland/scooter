package rknn

/*
#include <stdint.h>

typedef int32_t (*fn_rknn_outputs_get)(uint64_t ctx, uint32_t n_outputs, void* outputs, void* extend);
typedef int32_t (*fn_rknn_outputs_release)(uint64_t ctx, uint32_t n_outputs, void* outputs);

static inline int32_t call_outputs_get(uintptr_t fn, uint64_t ctx, uint32_t n, void* out) {
	return ((fn_rknn_outputs_get)fn)(ctx, n, out, 0);
}

static inline int32_t call_outputs_release(uintptr_t fn, uint64_t ctx, uint32_t n, void* out) {
	return ((fn_rknn_outputs_release)fn)(ctx, n, out);
}
*/
import "C"
import "unsafe"

func callRKNNOutputsGet(handle Context, nOutputs uint32, out *Output) int32 {
	return int32(C.call_outputs_get(C.uintptr_t(rknnOutputsGetSym), C.uint64_t(handle), C.uint32_t(nOutputs), unsafe.Pointer(out)))
}

func callRKNNOutputsRelease(handle Context, nOutputs uint32, out *Output) int32 {
	return int32(C.call_outputs_release(C.uintptr_t(rknnOutputsReleaseSym), C.uint64_t(handle), C.uint32_t(nOutputs), unsafe.Pointer(out)))
}
