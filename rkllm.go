package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"unsafe"

	"github.com/ebitengine/purego"
)

type LLMHandle unsafe.Pointer
type LLMCallState int32

const (
	RKLLM_RUN_NORMAL LLMCallState = 0
	RKLLM_RUN_WAITING             = 1
	RKLLM_RUN_FINISH              = 2
	RKLLM_RUN_ERROR               = 3
)

type RKLLMInputType int32

const (
	RKLLM_INPUT_PROMPT RKLLMInputType = 0
	RKLLM_INPUT_TOKEN  RKLLMInputType = 1
)

type RKLLMInferMode int32

const (
	RKLLM_INFER_GENERATE RKLLMInferMode = 0
)

type RKLLMResult struct {
	Text        *byte
	TokenID     int32
	PerfContext unsafe.Pointer
}

type RKLLMParam struct {
	ModelPath     *byte
	MaxContextLen int32
	MaxNewTokens  int32
	TopK          int32
	TopP          float32
	Temperature   float32
	RepeatPenalty float32
	FrequencyPen  float32
	PresencePen   float32
	Mirostat      int32
	MirostatTau   float32
	MirostatEta   float32
	LogitBias     unsafe.Pointer
	LogitBiasLen  int32
	IsAsync       bool
	ExtendParam   [128]byte
}

type RKLLMInput struct {
	Role           *byte
	EnableThinking bool
	InputType      RKLLMInputType
	PromptInput    *byte
}

type RKLLMCallback struct {
	Callback functionPointer
	Userdata unsafe.Pointer
}

type RKLLMInferParam struct {
	Mode               RKLLMInferMode
	LoraParams         unsafe.Pointer
	PromptCacheParams  unsafe.Pointer
	ModelParams        unsafe.Pointer
}

type functionPointer uintptr

func goStringToBytePtr(s string) *byte {
	b := append([]byte(s), 0)
	return &b[0]
}

func cStringToGo(ptr *byte) string {
	if ptr == nil {
		return ""
	}
	var length int
	for *(*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) + uintptr(length))) != 0 {
		length++
	}
	return string(unsafe.Slice(ptr, length))
}

func getRKLLMLibPath() string {
	if path := os.Getenv("RKLLM_LIB"); path != "" {
		return path
	}
	return "librkllmrt.so"
}

func getRKLLMModelPath(modelName string) string {
	if path := os.Getenv("RKLLM_MODEL"); path != "" {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	if modelName != "" && (filepath.IsAbs(modelName) || strings.Contains(modelName, "/") || strings.HasSuffix(modelName, ".rkllm")) {
		if _, err := os.Stat(modelName); err == nil {
			return modelName
		}
	}

	baseDir := "/home/scot/rkllama-models"
	target := strings.ToLower(modelName)
	if target == "" {
		target = "qwen3-4b"
	}
	if !strings.HasSuffix(target, ".rkllm") {
		target += ".rkllm"
	}

	var foundPath string
	_ = filepath.WalkDir(baseDir, func(path string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			name := strings.ToLower(d.Name())
			if name == target || strings.Contains(strings.ToLower(path), strings.ToLower(modelName)) {
				if strings.HasSuffix(name, ".rkllm") {
					foundPath = path
					return filepath.SkipAll
				}
			}
		}
		return nil
	})

	if foundPath != "" {
		return foundPath
	}

	matches, _ := filepath.Glob("*.rkllm")
	if len(matches) > 0 {
		return matches[0]
	}

	if modelName != "" {
		return modelName
	}
	return DefaultModelName
}

type rkllmSession struct {
	mu           sync.Mutex
	lib          uintptr
	handle       LLMHandle
	modelPath    string
	libPath      string
	initialized  bool
	rkllmRun     func(handle LLMHandle, input *RKLLMInput, inferParams *RKLLMInferParam, userdata unsafe.Pointer) int32
	rkllmDestroy func(handle LLMHandle) int32

	// Active request state
	activeChan    chan<- StreamMessage
	activeBuilder *strings.Builder
	activeParser  *RkToolParser
	lastStderr    string
	justLoaded    bool
}

var globalSession rkllmSession

func closeRKLLMSession() {
	globalSession.mu.Lock()
	defer globalSession.mu.Unlock()

	if globalSession.initialized {
		if globalSession.rkllmDestroy != nil && globalSession.handle != nil {
			globalSession.rkllmDestroy(globalSession.handle)
		}
		globalSession.initialized = false
		globalSession.handle = nil
		globalSession.lib = 0
	}
}

func captureStderr(fn func() error) (string, error) {
	r, w, err := os.Pipe()
	if err != nil {
		return "", fn()
	}

	oldStderrFd, err := syscall.Dup(2)
	if err != nil {
		r.Close()
		w.Close()
		return "", fn()
	}

	syscall.Dup3(int(w.Fd()), 2, 0)

	outChan := make(chan string, 1)
	go func() {
		var buf bytes.Buffer
		io.Copy(&buf, r)
		outChan <- buf.String()
	}()

	fnErr := fn()

	w.Close()
	syscall.Dup3(oldStderrFd, 2, 0)
	syscall.Close(oldStderrFd)
	r.Close()

	captured := <-outChan
	return strings.TrimSpace(captured), fnErr
}

func ensureSession(modelName string) error {
	libPath := getRKLLMLibPath()
	modelPath := getRKLLMModelPath(modelName)

	if globalSession.initialized && globalSession.modelPath == modelPath && globalSession.libPath == libPath {
		globalSession.justLoaded = false
		return nil
	}

	if globalSession.initialized {
		if globalSession.rkllmDestroy != nil && globalSession.handle != nil {
			globalSession.rkllmDestroy(globalSession.handle)
		}
		globalSession.initialized = false
		globalSession.handle = nil
	}

	var lib uintptr
	var handle LLMHandle
	var rkllmCreateDefaultParam func() RKLLMParam
	var rkllmInit func(handle *LLMHandle, param *RKLLMParam, callback *RKLLMCallback) int32
	var rkllmRun func(handle LLMHandle, input *RKLLMInput, inferParams *RKLLMInferParam, userdata unsafe.Pointer) int32
	var rkllmDestroy func(handle LLMHandle) int32

	capturedStderr, err := captureStderr(func() error {
		var dlErr error
		lib, dlErr = purego.Dlopen(libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if dlErr != nil {
			return dlErr
		}

		purego.RegisterLibFunc(&rkllmCreateDefaultParam, lib, "rkllm_createDefaultParam")
		purego.RegisterLibFunc(&rkllmInit, lib, "rkllm_init")
		purego.RegisterLibFunc(&rkllmRun, lib, "rkllm_run")
		purego.RegisterLibFunc(&rkllmDestroy, lib, "rkllm_destroy")

		callbackFn := func(result *RKLLMResult, userdata unsafe.Pointer, state LLMCallState) uintptr {
			globalSession.mu.Lock()
			ch := globalSession.activeChan
			builder := globalSession.activeBuilder
			parser := globalSession.activeParser
			globalSession.mu.Unlock()

			switch state {
			case RKLLM_RUN_NORMAL:
				if result != nil && result.Text != nil {
					text := cStringToGo(result.Text)
					if builder != nil {
						builder.WriteString(text)
					}
					if ch != nil && parser != nil {
						cleanText, calls := parser.ProcessChunk(text)
						select {
						case <-context.Background().Done():
						default:
							ch <- StreamMessage{
								Content:   cleanText,
								ToolCalls: calls,
								Done:      false,
							}
						}
					}
				}
			case RKLLM_RUN_FINISH:
				if ch != nil {
					ch <- StreamMessage{Done: true}
				}
			case RKLLM_RUN_ERROR:
				if ch != nil {
					ch <- StreamMessage{Err: fmt.Errorf("rkllm execution error"), Done: true}
				}
			}
			return 0
		}

		callbackPtr := purego.NewCallback(callbackFn)
		cbStruct := RKLLMCallback{
			Callback: functionPointer(callbackPtr),
			Userdata: nil,
		}

		param := rkllmCreateDefaultParam()
		param.ModelPath = goStringToBytePtr(modelPath)
		param.MaxContextLen = 4096
		param.MaxNewTokens = 512

		if ret := rkllmInit(&handle, &param, &cbStruct); ret != 0 {
			return fmt.Errorf("rkllm_init failed (model: %s): %d", modelPath, ret)
		}
		return nil
	})

	if err != nil {
		if capturedStderr != "" {
			return fmt.Errorf("%w\nStderr logs:\n%s", err, capturedStderr)
		}
		return err
	}

	globalSession.lib = lib
	globalSession.handle = handle
	globalSession.modelPath = modelPath
	globalSession.libPath = libPath
	globalSession.rkllmRun = rkllmRun
	globalSession.rkllmDestroy = rkllmDestroy
	globalSession.initialized = true
	globalSession.lastStderr = capturedStderr
	globalSession.justLoaded = true

	return nil
}

func streamRKLLM(ctx context.Context, modelName string, history []Message, channel chan<- StreamMessage) {
	globalSession.mu.Lock()
	if err := ensureSession(modelName); err != nil {
		globalSession.mu.Unlock()
		channel <- StreamMessage{Err: err, Done: true}
		return
	}

	justLoaded := globalSession.justLoaded
	lastStderr := globalSession.lastStderr
	globalSession.justLoaded = false

	rkParser := NewRkToolParser()
	globalSession.activeChan = channel
	globalSession.activeBuilder = nil
	globalSession.activeParser = rkParser
	runFn := globalSession.rkllmRun
	handle := globalSession.handle
	globalSession.mu.Unlock()

	// If the model was just loaded and emitted stderr logs, send them as a system message first
	if justLoaded && lastStderr != "" {
		channel <- StreamMessage{
			Content: fmt.Sprintf("[Runtime Stderr Logs during initialization]:\n%s\n\n", lastStderr),
			Done:    false,
		}
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var systemParts []string
	var prompt strings.Builder

	for _, msg := range history {
		switch msg.Role {
		case "system":
			systemParts = append(systemParts, msg.Content)
		case "user":
			prompt.WriteString("User: " + msg.Content + "\n")
		case "assistant":
			prompt.WriteString("Assistant: " + msg.Content + "\n")
		case "tool":
			prompt.WriteString("Tool result: " + msg.Content + "\n")
		}
	}
	systemParts = append(systemParts, buildToolInstructions(getSystemTools()))
	prompt.WriteString("Assistant: ")

	fullPrompt := "System:\n" + strings.Join(systemParts, "\n\n") + "\n\n" + prompt.String()

	inferParams := RKLLMInferParam{
		Mode: RKLLM_INFER_GENERATE,
	}

	input := RKLLMInput{
		InputType:   RKLLM_INPUT_PROMPT,
		PromptInput: goStringToBytePtr(fullPrompt),
	}

	if ret := runFn(handle, &input, &inferParams, nil); ret != 0 {
		channel <- StreamMessage{Err: fmt.Errorf("rkllm_run failed: %d", ret), Done: true}
	}
}

func runRKLLMSummary(modelName string, messages []Message) (string, error) {
	globalSession.mu.Lock()
	if err := ensureSession(modelName); err != nil {
		globalSession.mu.Unlock()
		return "", err
	}

	var sb strings.Builder
	globalSession.activeChan = nil
	globalSession.activeBuilder = &sb
	globalSession.activeParser = nil
	runFn := globalSession.rkllmRun
	handle := globalSession.handle
	globalSession.mu.Unlock()

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	prompt := "System:\nSummarize the conversation for a future assistant. Preserve user goals, decisions, changed files, relevant tool results, and unresolved work. Omit chatter. Return only the summary.\n\n" + renderMessages(messages) + "\n\nAssistant: "

	inferParams := RKLLMInferParam{
		Mode: RKLLM_INFER_GENERATE,
	}

	input := RKLLMInput{
		InputType:   RKLLM_INPUT_PROMPT,
		PromptInput: goStringToBytePtr(prompt),
	}

	if ret := runFn(handle, &input, &inferParams, nil); ret != 0 {
		return "", fmt.Errorf("rkllm_run failed: %d", ret)
	}

	return sb.String(), nil
}

func checkRKLLMStatus() string {
	libPath := getRKLLMLibPath()
	modelPath := getRKLLMModelPath("")
	globalSession.mu.Lock()
	loaded := globalSession.initialized
	stderr := globalSession.lastStderr
	globalSession.mu.Unlock()

	status := "Loaded in NPU memory"
	if !loaded {
		status = "Not currently loaded"
	}

	res := fmt.Sprintf("RKLLM runtime OK (lib: %s, model: %s, status: %s)", libPath, modelPath, status)
	if stderr != "" {
		res += fmt.Sprintf("\nLast Stderr logs:\n%s", stderr)
	}
	return res
}

func listRKLLMModels() []string {
	var models []string
	seen := make(map[string]bool)

	baseDir := "/home/scot/rkllama-models"
	_ = filepath.WalkDir(baseDir, func(path string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(strings.ToLower(d.Name()), ".rkllm") {
			if !seen[path] {
				seen[path] = true
				models = append(models, path)
			}
		}
		return nil
	})

	matches, _ := filepath.Glob("*.rkllm")
	for _, m := range matches {
		abs, err := filepath.Abs(m)
		if err == nil && !seen[abs] {
			seen[abs] = true
			models = append(models, abs)
		} else if !seen[m] {
			seen[m] = true
			models = append(models, m)
		}
	}

	if envModel := os.Getenv("RKLLM_MODEL"); envModel != "" {
		if !seen[envModel] {
			seen[envModel] = true
			models = append(models, envModel)
		}
	}

	if len(models) == 0 {
		models = append(models, "No .rkllm models found in "+baseDir)
	}
	return models
}
