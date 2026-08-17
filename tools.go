package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
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
						"file_path":    {Type: "string", Description: "Absolute output file path location."},
						"content": {Type: "string", Description: "Raw text or code contents for the file."},
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
	}
}

type WriteFileArgs struct {
	FilePath    string `json:"file_path"`
	Content string `json:"content"`
}

type SetDeviceStateArgs struct {
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
			} else {
				/* fmt.Fprintf(&sb, "%s\t%s\t\t\n", v.Alias, k)
				for _, c := range v.Children {
					fmt.Fprintf(&sb, "%s/%s\t%s%s\t\t%d\t\n", v.Alias, c.Alias, v.DeviceID, c.ID, c.RelayState)
				} */
				continue	
			}
		}
		return sb.String()
	case "set_device_state":
		var args SetDeviceStateArgs
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return fmt.Sprintf("System Error parsing arguments: %v", err)
		}
		
		state := 0
		if strings.ToLower(args.State) == "on" {
			state = 1
		} else if strings.ToLower(args.State) == "off" {
			state = 0
		} else {
			return "System Error: Invalid state, use 'on' or 'off'."
		}
		
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
	
		k, _ := kasa.NewDevice(args.IP)	
		if err := k.SetRelayStateCtx(ctx, state == 1); err != nil {
			return fmt.Sprintf("System Error setting device state: %v", err)
		}
		
		return fmt.Sprintf("Success: Set device %s to %s.", args.IP, args.State)
	}
	logDebug("dispatchToolCall failed for: %s", name)
	return fmt.Sprintf("System Error: Tool %s not implemented.", name)
}

