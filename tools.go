package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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

func dispatchToolCall(name string, rawArgs json.RawMessage) string {
	logDebug("dispatchToolCall called: %s, Args: %s", name, string(rawArgs))
	switch name {
	case "write_file":
		var args WriteFileArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}

		// Ensure target directory tree exists prior to writing
		dir := filepath.Dir(args.FilePath)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Sprintf("System Error creating directory %s: %v", dir, err)
		}

		if err := os.WriteFile(args.FilePath, []byte(args.Content), 0644); err != nil {
			return fmt.Sprintf("System Error writing path: %v", err)
		}
		res := fmt.Sprintf("Success: Intercepted text stream and wrote file directly to %s", args.FilePath)
		logDebug("dispatchToolCall success: %s", res)
		return res

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
