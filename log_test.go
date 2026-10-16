package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLogDebugWritesToCustomFile(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "log_test_*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	logFilePath := filepath.Join(tmpDir, "test.log")

	origLogFile := os.Getenv("SCOOTER_LOG_FILE")
	origDebug := os.Getenv("SCOOTER_DEBUG")
	defer func() {
		closeLogger()
		os.Setenv("SCOOTER_LOG_FILE", origLogFile)
		os.Setenv("SCOOTER_DEBUG", origDebug)
	}()

	closeLogger()
	os.Setenv("SCOOTER_LOG_FILE", logFilePath)
	os.Unsetenv("SCOOTER_DEBUG")

	testMsg := "hello scooter logging verification 987"
	logDebug("%s", testMsg)
	closeLogger()

	content, err := os.ReadFile(logFilePath)
	if err != nil {
		t.Fatalf("failed reading log file: %v", err)
	}

	if !strings.Contains(string(content), testMsg) {
		t.Fatalf("log content = %q, want it to contain %q", string(content), testMsg)
	}
}

func TestLogDebugDisabledViaEnv(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "log_test_*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	logFilePath := filepath.Join(tmpDir, "disabled.log")

	origLogFile := os.Getenv("SCOOTER_LOG_FILE")
	origDebug := os.Getenv("SCOOTER_DEBUG")
	defer func() {
		closeLogger()
		os.Setenv("SCOOTER_LOG_FILE", origLogFile)
		os.Setenv("SCOOTER_DEBUG", origDebug)
	}()

	closeLogger()
	os.Setenv("SCOOTER_LOG_FILE", logFilePath)
	os.Setenv("SCOOTER_DEBUG", "0")

	logDebug("this should never be written")
	closeLogger()

	if _, err := os.Stat(logFilePath); !os.IsNotExist(err) {
		t.Fatalf("expected log file to not exist when SCOOTER_DEBUG=0, stat err = %v", err)
	}
}
