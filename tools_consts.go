package main

import (
	"fmt"
	"strings"
)

var binaryFileExts = map[string]bool{
	".rkllm": true,
	".so":    true,
	".dll":   true,
	".dylib": true,
	".bin":   true,
	".exe":   true,
	".o":     true,
	".a":     true,
	".png":   true,
	".jpg":   true,
	".jpeg":  true,
	".gif":   true,
	".webp":  true,
	".ico":   true,
	".zip":   true,
	".tar":   true,
	".gz":    true,
	".7z":    true,
	".pdf":   true,
	".woff":  true,
	".woff2": true,
	".ttf":   true,
	".eot":   true,
	".mp3":   true,
	".mp4":   true,
	".wav":   true,
}

type Property struct {
	Type        string   `json:"type"`
	Description string   `json:"description"`
	Enum        []string `json:"enum,omitempty"`
}

type Parameters struct {
	Type       string              `json:"type"`
	Properties map[string]Property `json:"properties"`
	Required   []string            `json:"required"`
}

type FunctionDefinition struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Parameters  Parameters `json:"parameters"`
}

type Tool struct {
	Type     string             `json:"type"`
	Function FunctionDefinition `json:"function"`
}

var activeToolset = "all"

func GetActiveToolset() string {
	if activeToolset == "" {
		return "all"
	}
	return activeToolset
}

func SetActiveToolset(name string) error {
	normalized := strings.ToLower(strings.TrimSpace(name))
	switch normalized {
	case "dev", "development":
		activeToolset = "development"
	case "web", "net", "internet":
		activeToolset = "web"
	case "all", "every":
		activeToolset = "all"
	default:
		return fmt.Errorf("unknown toolset %q (available: development, web, all)", name)
	}
	return nil
}

func listToolsetsNames() []string {
	return []string{
		"development (dev)",
		"web (internet)",
		"all (default)",
	}
}

func getWebTools() []Tool {
	return []Tool{
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "web_fetch",
				Description: "Processes content from URL(s) embedded in a prompt. Include up to 20 URLs and instructions (e.g., summarize, extract specific data) directly in the 'prompt' parameter OR a single URL via the url parameter",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"prompt": {
							Type:        "string",
							Description: "A prompt containing URL(s) (up to 20) to fetch and instructions on how to process their content. URLs must start with http:// or https://.",
						},
						"url": {
							Type:        "string",
							Description: "A single URL to fetch",
						},
					},
				},
			},
		},
	}
}

func getDevelopmentTools() []Tool {
	return []Tool{
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "apply_patch",
				Description: "Applies a structured patch to files within the working directory.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"patch": {Type: "string", Description: "Patch text using Begin Patch/End Patch markers."},
					},
					Required: []string{"patch"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "git_status",
				Description: "Shows the Git working tree status for the working directory.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"path": {Type: "string", Description: "Optional path inside the working directory to limit the status query."},
					},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "git_diff",
				Description: "Shows the Git diff for the working directory or one file.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"path": {Type: "string", Description: "Optional path inside the working directory to limit the diff."},
					},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "find_files",
				Description: "Finds files recursively under the working directory using a glob pattern.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"pattern":     {Type: "string", Description: "Glob pattern such as '*.go' or '**/*_test.go'."},
						"max_results": {Type: "integer", Description: "Optional maximum number of paths to return; defaults to 100."},
					},
					Required: []string{"pattern"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "replace_text",
				Description: "Replaces exact text in a file within the working directory. Fails unless the expected match count is met.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"file_path":        {Type: "string", Description: "Path of the file to edit."},
						"old_text":         {Type: "string", Description: "Exact text to replace."},
						"new_text":         {Type: "string", Description: "Replacement text."},
						"expected_matches": {Type: "integer", Description: "Expected number of exact matches; defaults to 1."},
					},
					Required: []string{"file_path", "old_text", "new_text"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "file_metadata",
				Description: "Retrieves metadata for a file or directory (size, modification time, permissions, type).",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"file_path": {Type: "string", Description: "Path of the file or directory to inspect."},
					},
					Required: []string{"file_path"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "edit_file",
				Description: "Edits a file using structured operations (replace, insert, delete) with diff preview and atomic writes.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"file_path":    {Type: "string", Description: "Path of the file to edit."},
						"operations":   {Type: "array", Description: "List of edit operations (replace, insert, delete)."},
						"preview_only": {Type: "boolean", Description: "Optional: if true, returns diff preview without writing."},
					},
					Required: []string{"file_path", "operations"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "shell",
				Description: "Runs a shell command as the current user in the working directory. Use only when a local shell command is necessary.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"command": {Type: "string", Description: "Shell command to execute."},
					},
					Required: []string{"command"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "write_file",
				Description: "Writes text contents directly to a file on disk instead of outputting chat text.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"file_path": {Type: "string", Description: "Path of the file to write (relative to working directory or absolute within it)."},
						"content":   {Type: "string", Description: "Raw text or code contents for the file."},
					},
					Required: []string{"file_path", "content"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "read_file",
				Description: "Reads the content of a file on disk.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"file_path":  {Type: "string", Description: "Path of the file to read (relative to working directory or absolute within it)."},
						"start_line": {Type: "integer", Description: "Optional 1-based first line to read."},
						"end_line":   {Type: "integer", Description: "Optional inclusive last line to read."},
					},
					Required: []string{"file_path"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "list_directory",
				Description: "Lists files and subdirectories in a given directory.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"dir_path": {Type: "string", Description: "Directory path to list (relative to working directory or absolute within it; '.' for current directory)."},
					},
					Required: []string{"dir_path"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "outsource",
				Description: "Outsources a complex task or prompt to Google's Gemini GenAI API when the local model needs advanced cloud capabilities.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"prompt": {Type: "string", Description: "The prompt or instructions to send to Google GenAI."},
					},
					Required: []string{"prompt"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "grep_search",
				Description: "Recursively searches for a regex pattern in files within the current directory.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"pattern":     {Type: "string", Description: "The regex pattern to search for."},
						"max_results": {Type: "integer", Description: "Optional maximum number of matching lines to return; defaults to 50."},
					},
					Required: []string{"pattern"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "search_knowledge",
				Description: "Searches the local knowledge base for relevant content.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"query":      {Type: "string", Description: "Search query or keywords to find matching content."},
						"collection": {Type: "string", Description: "Optional collection or series filter."},
						"limit":      {Type: "integer", Description: "Optional maximum number of results to return (defaults to 5)."},
					},
					Required: []string{"query"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "ingest_knowledge",
				Description: "Ingests a local file into the knowledge base so it can be searched and analyzed later.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"file_path":  {Type: "string", Description: "Path to the file to ingest."},
						"collection": {Type: "string", Description: "Optional collection or series name."},
					},
					Required: []string{"file_path"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "search_knowledge_hybrid",
				Description: "Searches the knowledge base using semantic vector similarity and keyword search combined.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"query":      {Type: "string", Description: "Natural language query or concept."},
						"collection": {Type: "string", Description: "Optional collection or series filter."},
						"limit":      {Type: "integer", Description: "Max results to return (default 5)."},
					},
					Required: []string{"query"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "list_collections",
				Description: "Lists all available document collections and sermon series in the knowledge base.",
				Parameters: Parameters{
					Type:       "object",
					Properties: map[string]Property{},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "query_graph",
				Description: "Queries outgoing relationships and connected entities for a graph node (e.g., scripture references or metaphors tied to a sermon or preacher).",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"source_id": {Type: "string", Description: "ID of the root node (e.g., 'preacher_scot' or 'passage_exodus_16_1')."},
						"relation":  {Type: "string", Description: "Optional relation filter ('EXPOUNDS', 'USES_METAPHOR', 'PREACHED')."},
					},
					Required: []string{"source_id"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "reembed_knowledge",
				Description: "Triggers re-embedding of text chunks in the knowledge base using a specified embedding model.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"model_name": {Type: "string", Description: "Name of the embedding model being used."},
					},
					Required: []string{"model_name"},
				},
			},
		},
	}
}

func getSystemTools() []Tool {
	switch GetActiveToolset() {
	case "development":
		return getDevelopmentTools()
	case "web":
		return getWebTools()
	case "all":
		fallthrough
	default:
		return append(getDevelopmentTools(), getWebTools()...)
	}
}

type WebFetchArgs struct {
	Prompt string `json:"prompt"`
	URL    string `json:"url"`
}

type WriteFileArgs struct {
	FilePath string `json:"file_path"`
	Content  string `json:"content"`
}

type ShellArgs struct {
	Command string `json:"command"`
}

type ReplaceTextArgs struct {
	FilePath        string `json:"file_path"`
	OldText         string `json:"old_text"`
	NewText         string `json:"new_text"`
	ExpectedMatches int    `json:"expected_matches,omitempty"`
}

type FileMetadataArgs struct {
	FilePath string `json:"file_path"`
}

type EditOperation struct {
	Type        string      `json:"type"`
	Pattern     string      `json:"pattern,omitempty"`
	Replacement string      `json:"replacement,omitempty"`
	Flags       int         `json:"flags,omitempty"`
	Count       int         `json:"count,omitempty"`
	LineNumber  int         `json:"line_number,omitempty"`
	Content     interface{} `json:"content,omitempty"`
	StartLine   int         `json:"start_line,omitempty"`
	EndLine     int         `json:"end_line,omitempty"`
}

type EditFileArgs struct {
	FilePath    string          `json:"file_path"`
	Operations  []EditOperation `json:"operations"`
	PreviewOnly bool            `json:"preview_only,omitempty"`
}

type FindFilesArgs struct {
	Pattern    string `json:"pattern"`
	MaxResults int    `json:"max_results,omitempty"`
}

type GitPathArgs struct {
	Path string `json:"path,omitempty"`
}

type ApplyPatchArgs struct {
	Patch string `json:"patch"`
}

type ReadFileArgs struct {
	FilePath  string `json:"file_path"`
	StartLine int    `json:"start_line,omitempty"`
	EndLine   int    `json:"end_line,omitempty"`
}

type ListDirArgs struct {
	DirPath string `json:"dir_path"`
}

type GrepSearchArgs struct {
	Pattern    string `json:"pattern"`
	MaxResults int    `json:"max_results,omitempty"`
}

type OutsourceArgs struct {
	Prompt string `json:"prompt"`
}

type SetDeviceStateArgs struct {
	IP    string `json:"ip"`
	State string `json:"state"`
}

type SetBrightnessArgs struct {
	IP         string      `json:"ip"`
	Brightness FlexibleInt `json:"brightness"`
}

type DeviceIPOnlyArgs struct {
	IP string `json:"ip"`
}

type SearchKnowledgeArgs struct {
	Query      string `json:"query"`
	Collection string `json:"collection,omitempty"`
	Limit      int    `json:"limit,omitempty"`
}

type IngestKnowledgeArgs struct {
	FilePath   string `json:"file_path"`
	Collection string `json:"collection,omitempty"`
}

type SearchKnowledgeHybridArgs struct {
	Query      string `json:"query"`
	Collection string `json:"collection,omitempty"`
	Limit      int    `json:"limit,omitempty"`
}

type QueryGraphArgs struct {
	SourceID string `json:"source_id"`
	Relation string `json:"relation,omitempty"`
}

type ReembedKnowledgeArgs struct {
	ModelName string `json:"model_name"`
}

type ListCollectionsArgs struct{}
