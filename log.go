package main

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
)

var (
	logMu   sync.Mutex
	logFile *os.File
	logInit bool
	logOff  bool
)

func initLoggerLocked() {
	debugEnv := strings.TrimSpace(os.Getenv("SCOOTER_DEBUG"))
	if debugEnv == "0" || strings.EqualFold(debugEnv, "false") || strings.EqualFold(debugEnv, "off") || strings.EqualFold(debugEnv, "no") {
		logOff = true
		return
	}

	logPath := strings.TrimSpace(os.Getenv("SCOOTER_LOG_FILE"))
	if logPath == "" {
		logPath = "/tmp/scooter.log"
	} else if logPath == "/dev/null" || strings.EqualFold(logPath, "off") {
		logOff = true
		return
	}

	f, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		logOff = true
		return
	}
	logFile = f
	logOff = false
}

func logDebug(format string, a ...interface{}) {
	logMu.Lock()
	defer logMu.Unlock()

	if !logInit {
		logInit = true
		initLoggerLocked()
	}
	if logOff || logFile == nil {
		return
	}

	timestamp := time.Now().Format("2006-01-02 15:04:05")
	msg := fmt.Sprintf(format, a...)
	fmt.Fprintf(logFile, "[%s] %s\n", timestamp, msg)
}

func closeLogger() {
	logMu.Lock()
	defer logMu.Unlock()

	if logFile != nil {
		_ = logFile.Close()
		logFile = nil
	}
	logInit = false
	logOff = false
}
