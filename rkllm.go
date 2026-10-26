package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
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
	RKLLM_RUN_NORMAL  LLMCallState = 0
	RKLLM_RUN_WAITING              = 1
	RKLLM_RUN_FINISH               = 2
	RKLLM_RUN_ERROR                = 3
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

type RKLLMPromptCacheParam struct {
	SavePromptCache int32
	CachePath       *byte
}

type RKLLMInferParam struct {
	Mode              RKLLMInferMode
	LoraParams        unsafe.Pointer
	PromptCacheParams *RKLLMPromptCacheParam
	ModelParams       unsafe.Pointer
}

type functionPointer uintptr

// stringToBytePtr creates a null-terminated byte slice and returns a pointer
// to its first element alongside the backing slice so it can be pinned in scope.
func stringToBytePtr(s string) (*byte, []byte) {
	b := append([]byte(s), 0)
	return &b[0], b
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
	if cfgLib := loadConfig().RKLLMLib; cfgLib != "" {
		return cfgLib
	}
	return "librkllmrt.so"
}

func getRKLLMModelsDir() string {
	if dir := os.Getenv("RKLLM_MODELS_DIR"); dir != "" {
		return dir
	}
	if cfgDir := loadConfig().RKLLMModelsDir; cfgDir != "" {
		return cfgDir
	}
	if homeDir, err := os.UserHomeDir(); err == nil {
		return filepath.Join(homeDir, "rkllama-models")
	}
	return "rkllama-models"
}

func getDefaultModelName() string {
	if env := os.Getenv("RKLLM_MODEL"); env != "" {
		return env
	}
	if cfgModel := loadConfig().DefaultModel; cfgModel != "" {
		return getRKLLMModelPath(cfgModel)
	}
	return getRKLLMModelPath("qwen3-4b")
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

	baseDir := getRKLLMModelsDir()
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
	return "qwen3-4b"
}

type rkllmSession struct {
	inferMu              sync.Mutex // Serializes all NPU inferences & session init/teardown
	mu                   sync.Mutex // Protects metadata
	lib                  uintptr
	handle               LLMHandle
	modelPath            string
	libPath              string
	contextLen           int
	initialized          bool
	rkllmRun             func(handle LLMHandle, input *RKLLMInput, inferParams *RKLLMInferParam, userdata unsafe.Pointer) int32
	rkllmDestroy         func(handle LLMHandle) int32
	rkllmAbort           func(handle LLMHandle) int32
	rkllmLoadPromptCache func(handle LLMHandle, path *byte) int32

	// Active request state
	activeChan    chan<- StreamMessage
	activeBuilder *strings.Builder
	activeParser  *RkToolParser
	activeCtx     context.Context
	lastStderr    string
	justLoaded    bool
}

var globalSession rkllmSession

// Package-level static callback allocation to prevent executable memory leaks across session reloads.
var (
	globalCallbackPtr  uintptr
	globalCallbackOnce sync.Once
)

func getRKLLMCallback() RKLLMCallback {
	globalCallbackOnce.Do(func() {
		callbackFn := func(result *RKLLMResult, userdata unsafe.Pointer, state LLMCallState) uintptr {
			globalSession.mu.Lock()
			ch := globalSession.activeChan
			builder := globalSession.activeBuilder
			parser := globalSession.activeParser
			reqCtx := globalSession.activeCtx
			globalSession.mu.Unlock()

			if reqCtx == nil {
				reqCtx = context.Background()
			}

			switch state {
			case RKLLM_RUN_NORMAL:
				if ch == nil && builder == nil {
					return 1
				}
				if result != nil && result.Text != nil {
					text := cStringToGo(result.Text)
					if builder != nil {
						builder.WriteString(text)
					}
					if ch != nil && parser != nil {
						cleanText, calls := parser.ProcessChunk(text)
						if cleanText != "" || len(calls) > 0 {
							select {
							case <-reqCtx.Done():
								return 1
							case ch <- StreamMessage{
								Content:   cleanText,
								ToolCalls: calls,
								Done:      false,
							}:
							}
						}
					}
				}
			case RKLLM_RUN_FINISH:
				if ch != nil {
					select {
					case <-reqCtx.Done():
					case ch <- StreamMessage{Done: true}:
					}
				}
			case RKLLM_RUN_ERROR:
				if ch != nil {
					select {
					case <-reqCtx.Done():
					case ch <- StreamMessage{Err: fmt.Errorf("rkllm execution error"), Done: true}:
					}
				}
			}
			return 0
		}
		globalCallbackPtr = purego.NewCallback(callbackFn)
	})

	return RKLLMCallback{
		Callback: functionPointer(globalCallbackPtr),
		Userdata: nil,
	}
}

func abortRKLLM() {
	globalSession.mu.Lock()
	handle := globalSession.handle
	abortFn := globalSession.rkllmAbort
	isInit := globalSession.initialized
	globalSession.mu.Unlock()

	if isInit && abortFn != nil && handle != nil {
		abortFn(handle)
	}
}

func closeRKLLMSession() {
	globalSession.inferMu.Lock()
	defer globalSession.inferMu.Unlock()

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
	defer r.Close()

	oldStderrFd, err := syscall.Dup(2)
	if err != nil {
		w.Close()
		return "", fn()
	}

	outChan := make(chan string, 1)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)
		outChan <- buf.String()
	}()

	var fnErr error
	func() {
		defer func() {
			syscall.Dup3(oldStderrFd, 2, 0)
			syscall.Close(oldStderrFd)
			w.Close()
		}()
		syscall.Dup3(int(w.Fd()), 2, 0)
		fnErr = fn()
	}()

	captured := <-outChan
	return strings.TrimSpace(captured), fnErr
}

func ensureSession(modelName string, contextLen int) error {
	if contextLen <= 0 {
		contextLen = getActiveContextLimit()
	}
	libPath := getRKLLMLibPath()
	modelPath := getRKLLMModelPath(modelName)

	globalSession.mu.Lock()
	if globalSession.initialized && globalSession.modelPath == modelPath && globalSession.libPath == libPath && globalSession.contextLen == contextLen {
		globalSession.justLoaded = false
		globalSession.mu.Unlock()
		return nil
	}

	if globalSession.initialized {
		if globalSession.rkllmDestroy != nil && globalSession.handle != nil {
			globalSession.rkllmDestroy(globalSession.handle)
		}
		globalSession.initialized = false
		globalSession.handle = nil
	}
	lib := globalSession.lib
	globalSession.mu.Unlock()

	var handle LLMHandle
	var rkllmCreateDefaultParam func() RKLLMParam
	var rkllmInit func(handle *LLMHandle, param *RKLLMParam, callback *RKLLMCallback) int32
	var rkllmRun func(handle LLMHandle, input *RKLLMInput, inferParams *RKLLMInferParam, userdata unsafe.Pointer) int32
	var rkllmDestroy func(handle LLMHandle) int32
	var rkllmAbort func(handle LLMHandle) int32
	var rkllmLoadPromptCache func(handle LLMHandle, path *byte) int32

	capturedStderr, err := captureStderr(func() error {
		if lib == 0 {
			var dlErr error
			// Pure RTLD_NOW without RTLD_GLOBAL prevents symbol contamination
			lib, dlErr = purego.Dlopen(libPath, purego.RTLD_NOW)
			if dlErr != nil {
				return dlErr
			}
		}

		purego.RegisterLibFunc(&rkllmCreateDefaultParam, lib, "rkllm_createDefaultParam")
		purego.RegisterLibFunc(&rkllmInit, lib, "rkllm_init")
		purego.RegisterLibFunc(&rkllmRun, lib, "rkllm_run")
		purego.RegisterLibFunc(&rkllmDestroy, lib, "rkllm_destroy")
		purego.RegisterLibFunc(&rkllmAbort, lib, "rkllm_abort")

		cbStruct := getRKLLMCallback()

		modelPathPtr, modelPathSlice := stringToBytePtr(modelPath)
		defer runtime.KeepAlive(modelPathSlice)

		param := rkllmCreateDefaultParam()
		param.ModelPath = modelPathPtr
		param.MaxContextLen = int32(contextLen)

		// Set high generation upper bound to prevent truncating agent/tool responses
		param.MaxNewTokens = 4096

		// Enable all 3 NPU Cores on the RK3588 (Core 0 + Core 1 + Core 2 mask)
		param.ExtendParam[0] = 7

		purego.RegisterLibFunc(&rkllmLoadPromptCache, lib, "rkllm_load_prompt_cache")

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

	globalSession.mu.Lock()
	globalSession.lib = lib
	globalSession.handle = handle
	globalSession.modelPath = modelPath
	globalSession.libPath = libPath
	globalSession.contextLen = contextLen
	globalSession.rkllmRun = rkllmRun
	globalSession.rkllmDestroy = rkllmDestroy
	globalSession.rkllmAbort = rkllmAbort
	globalSession.rkllmLoadPromptCache = rkllmLoadPromptCache
	globalSession.initialized = true
	globalSession.lastStderr = capturedStderr
	globalSession.justLoaded = true
	globalSession.mu.Unlock()

	return nil
}

func getPromptCacheFile(modelPath string, contextLen int) string {
	if custom := loadConfig().RKLLMCachePath; custom != "" {
		return custom
	}

	if modelPath == "" {
		modelPath = getDefaultModelName()
	}
	if contextLen <= 0 {
		contextLen = getActiveContextLimit()
	}

	toolset := GetActiveToolset()

	baseName := filepath.Base(modelPath)
	baseName = strings.TrimSuffix(baseName, filepath.Ext(baseName))
	if baseName == "" {
		baseName = "default_model"
	}

	hashInput := fmt.Sprintf("%s|%s", modelPath, toolset)
	hash := sha256.Sum256([]byte(hashInput))
	shortHash := hex.EncodeToString(hash[:])[:8]

	cacheFileName := fmt.Sprintf("%s_%s_ctx%d_%s.rkllm_cache", baseName, toolset, contextLen, shortHash)

	if cacheDir, err := getCacheFile(); err == nil && cacheDir != "" {
		return filepath.Join(filepath.Dir(cacheDir), cacheFileName)
	}

	return cacheFileName
}

func getPromptCacheFileForSystemPrompt(modelPath string, contextLen int, systemPrompt string) string {
	basePath := getPromptCacheFile(modelPath, contextLen)
	hashInput := fmt.Sprintf("%s|%d|%s", modelPath, contextLen, systemPrompt)
	hash := sha256.Sum256([]byte(hashInput))
	suffix := hex.EncodeToString(hash[:])[:12]
	ext := filepath.Ext(basePath)
	base := strings.TrimSuffix(basePath, ext)
	return base + "_" + suffix + ext
}

func streamRKLLM(ctx context.Context, modelName string, history []Message, channel chan<- StreamMessage) {
	globalSession.inferMu.Lock()
	defer globalSession.inferMu.Unlock()

	contextLen := getActiveContextLimit()
	if err := ensureSession(modelName, contextLen); err != nil {
		channel <- StreamMessage{Err: err, Done: true}
		return
	}

	globalSession.mu.Lock()
	activeModelPath := globalSession.modelPath
	justLoaded := globalSession.justLoaded
	lastStderr := globalSession.lastStderr
	globalSession.justLoaded = false

	rkParser := NewRkToolParser()
	globalSession.activeChan = channel
	globalSession.activeBuilder = nil
	globalSession.activeParser = rkParser
	globalSession.activeCtx = ctx
	runFn := globalSession.rkllmRun
	loadPromptCacheFn := globalSession.rkllmLoadPromptCache
	handle := globalSession.handle
	globalSession.mu.Unlock()

	defer func() {
		globalSession.mu.Lock()
		globalSession.activeChan = nil
		globalSession.activeParser = nil
		globalSession.activeCtx = nil
		globalSession.mu.Unlock()
	}()

	if justLoaded && lastStderr != "" {
		select {
		case <-ctx.Done():
			channel <- StreamMessage{Err: ctx.Err(), Done: true}
			return
		case channel <- StreamMessage{
			Content: fmt.Sprintf("[Runtime Stderr Logs during initialization]:\n%s\n\n", lastStderr),
			Done:    false,
		}:
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
			if msg.Content != "" {
				prompt.WriteString("Assistant: " + msg.Content + "\n")
			}
			for _, call := range msg.ToolCalls {
				arguments := call.Function.Arguments
				if len(arguments) == 0 {
					arguments = json.RawMessage(`{}`)
				}
				payload, err := json.Marshal(struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				}{Name: call.Function.Name, Arguments: arguments})
				if err == nil {
					fmt.Fprintf(&prompt, "%s%s%s\n", toolCallStartTag, payload, toolCallEndTag)
				}
			}
		case "tool":
			if msg.ToolName != "" {
				fmt.Fprintf(&prompt, "Tool result from %s: %s\n", msg.ToolName, msg.Content)
			} else {
				prompt.WriteString("Tool result: " + msg.Content + "\n")
			}
		}
	}
	systemParts = append(systemParts, buildToolInstructions(getSystemTools()))
	prompt.WriteString("Assistant: ")

	inferParams := RKLLMInferParam{
		Mode: RKLLM_INFER_GENERATE,
	}

	cfg := loadConfig()
	var cacheParam RKLLMPromptCacheParam
	systemPrompt := "System:\n" + strings.Join(systemParts, "\n\n") + "\n\n"
	cacheFile := getPromptCacheFileForSystemPrompt(activeModelPath, contextLen, systemPrompt)
	// cacheLoaded := false

	if cfg.RKLLMCacheEnabled && loadPromptCacheFn != nil {
		if _, err := os.Stat(cacheFile); err != nil {
			if dir := filepath.Dir(cacheFile); dir != "." {
				if mkdirErr := os.MkdirAll(dir, 0755); mkdirErr != nil {
					fmt.Fprintf(os.Stderr, "Warning: create prompt cache directory: %v\n", mkdirErr)
				}
			}
			globalSession.mu.Lock()
			previousChan, previousParser, previousCtx := globalSession.activeChan, globalSession.activeParser, globalSession.activeCtx
			globalSession.activeChan, globalSession.activeParser, globalSession.activeCtx = nil, nil, nil
			globalSession.mu.Unlock()

			cacheErr := createSystemPromptCache(runFn, handle, cacheFile, systemPrompt)

			globalSession.mu.Lock()
			globalSession.activeChan, globalSession.activeParser, globalSession.activeCtx = previousChan, previousParser, previousCtx
			globalSession.mu.Unlock()
			if cacheErr != nil {
				fmt.Fprintf(os.Stderr, "Warning: %v\n", cacheErr)
			}
		}
		if _, err := os.Stat(cacheFile); err == nil {
			cachePathPtr, cachePathSlice := stringToBytePtr(cacheFile)
			defer runtime.KeepAlive(cachePathSlice)

			if ret := loadPromptCacheFn(handle, cachePathPtr); ret != 0 {
				fmt.Fprintf(os.Stderr, "Warning: rkllm_load_prompt_cache failed: %d\n", ret)
			} else {
				// cacheLoaded = true
				cacheParam.SavePromptCache = 0
				cacheParam.CachePath = cachePathPtr
				inferParams.PromptCacheParams = &cacheParam
			}
		}
	}

	/* librkllmrt.so prompt caching does NOT work by loading a static file into memory and letting you append arbitrary context on the fly without system wrappers.
In librkllmrt, rkllm_load_prompt_cache expects the incoming prompt in rkllm_run to contain the exact prefix matching the cached tokens, or the transformer position embeddings ($pos_0, pos_1, ...$  ) alignment breaks!
	*/
	/* var conversationPrompt string
	if cacheLoaded {
		conversationPrompt = prompt.String()
	} else {
		conversationPrompt = systemPrompt + prompt.String()
	} */
	// Always include systemPrompt in the text so positional embeddings align with the cache, 
	conversationPrompt := systemPrompt + prompt.String()

	promptPtr, promptSlice := stringToBytePtr(conversationPrompt)
	defer runtime.KeepAlive(promptSlice)

	input := RKLLMInput{
		InputType:   RKLLM_INPUT_PROMPT,
		PromptInput: promptPtr,
	}

	if ret := runFn(handle, &input, &inferParams, nil); ret != 0 {
		select {
		case <-ctx.Done():
		case channel <- StreamMessage{Err: fmt.Errorf("rkllm_run failed: %d", ret), Done: true}:
		}
	}
}

func runRKLLMSummary(modelName string, messages []Message) (string, error) {
	globalSession.inferMu.Lock()
	defer globalSession.inferMu.Unlock()

	contextLen := getActiveContextLimit()
	if err := ensureSession(modelName, contextLen); err != nil {
		return "", err
	}

	var sb strings.Builder
	globalSession.mu.Lock()
	globalSession.activeChan = nil
	globalSession.activeBuilder = &sb
	globalSession.activeParser = nil
	globalSession.activeCtx = context.Background()
	runFn := globalSession.rkllmRun
	handle := globalSession.handle
	globalSession.mu.Unlock()

	defer func() {
		globalSession.mu.Lock()
		globalSession.activeBuilder = nil
		globalSession.activeCtx = nil
		globalSession.mu.Unlock()
	}()

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	promptStr := "System:\nSummarize the conversation for a future assistant. Preserve user goals, decisions, changed files, relevant tool results, and unresolved work. Omit chatter. Return only the summary.\n\n" + renderMessages(messages) + "\n\nAssistant: "
	promptPtr, promptSlice := stringToBytePtr(promptStr)
	defer runtime.KeepAlive(promptSlice)

	inferParams := RKLLMInferParam{
		Mode: RKLLM_INFER_GENERATE,
	}

	input := RKLLMInput{
		InputType:   RKLLM_INPUT_PROMPT,
		PromptInput: promptPtr,
	}

	if ret := runFn(handle, &input, &inferParams, nil); ret != 0 {
		return "", fmt.Errorf("rkllm_run failed: %d", ret)
	}

	return sb.String(), nil
}

func checkRKLLMStatus(modelName string) string {
	libPath := getRKLLMLibPath()
	globalSession.mu.Lock()
	loaded := globalSession.initialized
	stderr := globalSession.lastStderr
	ctxLen := globalSession.contextLen
	modelPath := globalSession.modelPath
	globalSession.mu.Unlock()

	if modelPath == "" {
		modelPath = getRKLLMModelPath(modelName)
	}

	status := "Loaded in NPU memory"
	if !loaded {
		status = "Not currently loaded"
	}

	res := fmt.Sprintf("RKLLM runtime OK (lib: %s, model: %s, context: %d tokens, status: %s)", libPath, modelPath, ctxLen, status)
	if stderr != "" {
		res += fmt.Sprintf("\nLast Stderr logs:\n%s", stderr)
	}
	return res
}

func listRKLLMModels() []string {
	var models []string
	seen := make(map[string]bool)

	baseDir := getRKLLMModelsDir()
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

func createSystemPromptCache(
	runFn func(handle LLMHandle, input *RKLLMInput, inferParams *RKLLMInferParam, userdata unsafe.Pointer) int32,
	handle LLMHandle,
	systemCacheFile string,
	systemOnlyPrompt string,
) error {
	if _, err := os.Stat(systemCacheFile); err == nil {
		return nil
	}

	var cacheParam RKLLMPromptCacheParam
	cachePathPtr, cachePathSlice := stringToBytePtr(systemCacheFile)
	defer runtime.KeepAlive(cachePathSlice)

	cacheParam.CachePath = cachePathPtr
	cacheParam.SavePromptCache = 1

	inferParams := RKLLMInferParam{
		Mode:              RKLLM_INFER_GENERATE,
		PromptCacheParams: &cacheParam,
	}

	promptPtr, promptSlice := stringToBytePtr(systemOnlyPrompt)
	defer runtime.KeepAlive(promptSlice)

	input := RKLLMInput{
		InputType:   RKLLM_INPUT_PROMPT,
		PromptInput: promptPtr,
	}

	ret := runFn(handle, &input, &inferParams, nil)
	if _, err := os.Stat(systemCacheFile); err == nil {
		return nil
	}

	if ret != 0 {
		return fmt.Errorf("rkllm pre-caching failed with code %d", ret)
	}

	return nil
}
