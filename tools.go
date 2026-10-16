package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/cloudkucooland/go-kasa"
)

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

func getSystemTools() []Tool {
	return []Tool{
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "write_file",
				Description: "Writes text contents directly to a file on disk instead of outputting chat text.",
				Parameters: Parameters{
					Type: "object",
					Properties: map[string]Property{
						"file_path": {Type: "string", Description: "Absolute output file path location."},
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
						"file_path": {Type: "string", Description: "Absolute file path location to read."},
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
						"dir_path": {Type: "string", Description: "Absolute directory path to list."},
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
						"brightness": {Type: "string", Description: "Brightness level from 1 to 100."},
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
	IP         string `json:"ip"`
	Brightness string `json:"brightness"`
}

type DeviceIPOnlyArgs struct {
	IP string `json:"ip"`
}

type SetLEDStateArgs struct {
	IP    string `json:"ip"`
	State string `json:"state"`
}

func validatePath(path string) (string, error) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}

	rel, err := filepath.Rel(workingDir, absPath)
	if err != nil {
		return "", err
	}

	if strings.HasPrefix(rel, "..") || rel == ".." {
		return "", fmt.Errorf("path %s is outside of allowed directory %s", absPath, workingDir)
	}

	return absPath, nil
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

		cwd, err := os.Getwd()
		if err != nil {
			return fmt.Sprintf("System Error: Could not get working directory: %v", err)
		}

		var sb strings.Builder
		err = filepath.WalkDir(cwd, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				// Skip .git directory
				if d.Name() == ".git" {
					return filepath.SkipDir
				}
				return nil
			}

			content, err := os.ReadFile(path)
			if err != nil {
				return nil // Skip files we cannot read
			}

			if re.Match(content) {
				rel, _ := filepath.Rel(cwd, path)
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

		var brightness int
		if _, err := fmt.Sscanf(args.Brightness, "%d", &brightness); err != nil || brightness < 1 || brightness > 100 {
			return fmt.Sprintf("System Error: Brightness must be a number between 1 and 100, got '%s'", args.Brightness)
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
