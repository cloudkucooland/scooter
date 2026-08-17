package main

import (
	"fmt"
	"os"
	"time"
)

func logDebug(format string, a ...interface{}) {
	f, err := os.OpenFile("/tmp/scooter.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	fmt.Fprintf(f, "[%s] %s\n", timestamp, fmt.Sprintf(format, a...))
}
