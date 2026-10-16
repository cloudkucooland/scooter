package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/cloudkucooland/go-kasa"
)

const maxGrepFileSize = 2 * 1024 * 1024 // 2 MB

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

func getSystemTools() []Tool{
	return []Tool{
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
						"file_path": {Type: "string", Description: "Path of the file to read (relative to working directory or absolute within it)."},
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
				Name:        "grep_search",
				Description: "Recursively searches for a regex pattern in files within the current directory.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"pattern": {Type: "string", Description: "The regex pattern to search for."},
					},
					Required: []string{"pattern"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "get_devices",
				Description: "Reports all smart-home device status.",
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "set_device_state",
				Description: "Sets the state (on/off) of a smart-home device.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"ip":    {Type: "string", Description: "IP address of the device."},
						"state": {Type: "string", Description: "Desired state: 'on' or 'off'."},
					},
					Required: []string{"ip", "state"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "set_brightness",
				Description: "Sets the brightness of a dimmable smart light or switch (1-100).",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"ip":         {Type: "string", Description: "IP address of the device."},
						"brightness": {Type: "integer", Description: "Brightness level from 1 to 100."},
					},
					Required: []string{"ip", "brightness"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "get_sysinfo",
				Description: "Retrieves detailed system parameters and status from a specific device.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"ip": {Type: "string", Description: "IP address of the device."},
					},
					Required: []string{"ip"},
				},
			},
		},
	}
}

// FlexibleInt supports unmarshaling integers from JSON numeric literals or quoted strings.
type FlexibleInt int

func (fi *FlexibleInt) UnmarshalJSON(data []byte) error {
	data = bytes.TrimSpace(data)
	if len(data) == 0 || bytes.Equal(data, []byte("null")) {
		*fi = 0
		return nil
	}
	if data[0] == '"' {
		var s string
		if err := json.Unmarshal(data, &s); err != nil {
			return err
		}
		s = strings.TrimSpace(s)
		s = strings.TrimSuffix(s, "%")
		var val int
		if _, err := fmt.Sscanf(s, "%d", &val); err != nil {
			return fmt.Errorf("invalid integer: %q", s)
		}
		*fi = FlexibleInt(val)
		return nil
	}
	var val int
	if err := json.Unmarshal(data, &val); err != nil {
		return err
	}
	*fi = FlexibleInt(val)
	return nil
}

type WriteFileArgs struct {
	FilePath string `json:"file_path"`
	Content  string `json:"content"`
}

type ReadFileArgs struct {
	FilePath string `json:"file_path"`
}

type ListDirArgs struct {
	DirPath string `json:"dir_path"`
}

type GrepSearchArgs struct {
	Pattern string `json:"pattern"`
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

func validatePath(path string) (string, error) {
	baseDir := workingDir
	if baseDir == "" {
		if cwd, err := os.Getwd(); err == nil {
			baseDir, _ = filepath.Abs(cwd)
		}
	}

	var targetPath string
	if filepath.IsAbs(path) {
		targetPath = filepath.Clean(path)
	} else {
		targetPath = filepath.Join(baseDir, path)
	}

	rel, err := filepath.Rel(baseDir, targetPath)
	if err != nil {
		return "", err
	}

	if strings.HasPrefix(rel, "..") || rel == ".." {
		return "", fmt.Errorf("path %s is outside of allowed directory %s", targetPath, baseDir)
	}

	return targetPath, nil
}

func dispatchToolCall(name string, rawArgs json.RawMessage) string {
	logDebug("dispatchToolCall called: %s, Args: %s", name, string(rawArgs))
	switch name {
	case "write_file":
		var args WriteFileArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}

		absPath, err := validatePath(args.FilePath)
		if err != nil {
			return fmt.Sprintf("System Error: %v", err)
		}

		// Ensure target directory tree exists prior to writing
		dir := filepath.Dir(absPath)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Sprintf("System Error creating directory %s: %v", dir, err)
		}

		if err := os.WriteFile(absPath, []byte(args.Content), 0644); err != nil {
			return fmt.Sprintf("System Error writing path: %v", err)
		}
		res := fmt.Sprintf("Success: Intercepted text stream and wrote file directly to %s", absPath)
		logDebug("dispatchToolCall success: %s", res)
		return res

	case "read_file":
		var args ReadFileArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}

		absPath, err := validatePath(args.FilePath)
		if err != nil {
			return fmt.Sprintf("System Error: %v", err)
		}

		content, err := os.ReadFile(absPath)
		if err != nil {
			return fmt.Sprintf("System Error reading path: %v", err)
		}
		return string(content)

	case "list_directory":
		var args ListDirArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}

		absPath, err := validatePath(args.DirPath)
		if err != nil {
			return fmt.Sprintf("System Error: %v", err)
		}

		entries, err := os.ReadDir(absPath)
		if err != nil {
			return fmt.Sprintf("System Error listing directory: %v", err)
		}

		var sb strings.Builder
		for _, entry := range entries {
			if entry.IsDir() {
				fmt.Fprintf(&sb, "%s/\n", entry.Name())
			} else {
				fmt.Fprintf(&sb, "%s\n", entry.Name())
			}
		}
		return sb.String()

	case "grep_search":
		var args GrepSearchArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}

		re, err := regexp.Compile(args.Pattern)
		if err != nil {
			return fmt.Sprintf("System Error: Invalid regex pattern: %v", err)
		}

		baseDir := workingDir
		if baseDir == "" {
			if cwd, err := os.Getwd(); err == nil {
				baseDir, _ = filepath.Abs(cwd)
			}
		}

		var sb strings.Builder
		err = filepath.WalkDir(baseDir, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				// Skip .git directory and hidden dirs
				if d.Name() == ".git" {
					return filepath.SkipDir
				}
				return nil
			}

			// Skip non-regular files and files exceeding 2 MB limit
			info, err := d.Info()
			if err != nil || !info.Mode().IsRegular() || info.Size() > maxGrepFileSize {
				return nil
			}

			// Skip known binary extensions
			if binaryFileExts[strings.ToLower(filepath.Ext(path))] {
				return nil
			}

			f, err := os.Open(path)
			if err != nil {
				return nil // Skip files we cannot open
			}
			defer f.Close()

			// Check first 512 bytes for null byte (binary file indicator)
			header := make([]byte, 512)
			n, _ := f.Read(header)
			if bytes.IndexByte(header[:n], 0) != -1 {
				return nil
			}

			if _, err := f.Seek(0, io.SeekStart); err != nil {
				return nil
			}

			content, err := io.ReadAll(io.LimitReader(f, maxGrepFileSize))
			if err != nil {
				return nil
			}

			if re.Match(content) {
				rel, _ := filepath.Rel(baseDir, path)
				fmt.Fprintf(&sb, "%s\n", rel)
			}
			return nil
		})

		if err != nil {
			return fmt.Sprintf("System Error searching files: %v", err)
		}

		if sb.Len() == 0 {
			return "No matches found."
		}
		return sb.String()

	case "get_devices":
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		d, err := kasa.BroadcastDiscovery(ctx, 2)
		if err != nil {
			return fmt.Sprintf("System Error doing kasa discovery: %v", err)
		}

		var sb strings.Builder
		fmt.Fprintf(&sb, "Device\tIP/ID:\tState\tBrightness\n")

		for k, v := range d {
			if len(v.Children) == 0 {
				fmt.Fprintf(&sb, "%s\t%s\t%d\t\t%3d\n", v.Alias, k, v.RelayState, v.Brightness)
			}
			// restore child logic, but exceeds the context size
		}

		if sb.Len() == 0 {
			return "No Kasa smart home devices discovered on the local network."
		}

		return sb.String()

	case "set_device_state":
		var args SetDeviceStateArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}

		targetState := strings.ToLower(strings.TrimSpace(args.State))
		var turnOn bool
		if targetState == "on" || targetState == "1" || targetState == "true" {
			turnOn = true
		} else if targetState == "off" || targetState == "0" || targetState == "false" {
			turnOn = false
		} else {
			return fmt.Sprintf("System Error: Invalid state '%s', use 'on' or 'off'.", args.State)
		}

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		dev, err := kasa.NewDevice(args.IP)
		if err != nil || dev == nil {
			return fmt.Sprintf("System Error creating Kasa device instance for IP %s: %v", args.IP, err)
		}

		if err := dev.SetRelayStateCtx(ctx, turnOn); err != nil {
			return fmt.Sprintf("System Error setting device state: %v", err)
		}

		return fmt.Sprintf("Success: Set device %s to %s.", args.IP, targetState)

	case "set_brightness":
		var args SetBrightnessArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}

		brightness := int(args.Brightness)
		if brightness < 1 || brightness > 100 {
			return fmt.Sprintf("System Error: Brightness must be a number between 1 and 100, got %d", brightness)
		}

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		dev, err := kasa.NewDevice(args.IP)
		if err != nil || dev == nil {
			return fmt.Sprintf("System Error creating Kasa instance for IP %s: %v", args.IP, err)
		}

		if err := dev.SetBrightnessCtx(ctx, brightness); err != nil {
			return fmt.Sprintf("System Error setting brightness: %v", err)
		}

		return fmt.Sprintf("Success: Set brightness for %s to %d%%.", args.IP, brightness)

	case "get_sysinfo":
		var args DeviceIPOnlyArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		dev, err := kasa.NewDevice(args.IP)
		if err != nil || dev == nil {
			return fmt.Sprintf("System Error creating Kasa instance for IP %s: %v", args.IP, err)
		}

		sysInfo, err := dev.GetSettingsCtx(ctx)
		if err != nil {
			return fmt.Sprintf("System Error fetching sysinfo: %v", err)
		}

		infoJSON, _ := json.MarshalIndent(sysInfo, "", "  ")
		return string(infoJSON)
	}

	logDebug("dispatchToolCall failed for: %s", name)
	return fmt.Sprintf("System Error: Tool %s not implemented.", name)
}
