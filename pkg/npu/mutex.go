// Package npu provides a shared mutex for NPU access across rkllm and rknn.
package npu

import "sync"

var npuMu sync.Mutex

// Lock acquires the global NPU mutex.
func Lock() { npuMu.Lock() }

// Unlock releases the global NPU mutex.
func Unlock() { npuMu.Unlock() }
