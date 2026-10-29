package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTaskManagement(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "scooter_task_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	taskPath := filepath.Join(tmpDir, "tasks.json")
	t.Setenv("SCOOTER_TASK_PATH", taskPath)

	// Initial load should be empty
	tasks, err := LoadTasks()
	if err != nil {
		t.Fatalf("LoadTasks error = %v", err)
	}
	if len(tasks) != 0 {
		t.Fatalf("expected 0 initial tasks, got %d", len(tasks))
	}

	// Set tasks
	steps := []string{"Research codebase", "Implement feature", "Run tests"}
	if err := SetTasks(steps); err != nil {
		t.Fatalf("SetTasks error = %v", err)
	}

	tasks, err = LoadTasks()
	if err != nil || len(tasks) != 3 {
		t.Fatalf("expected 3 tasks after SetTasks, got %d (err: %v)", len(tasks), err)
	}

	// Add new task
	if err := AddTask("Review PR"); err != nil {
		t.Fatalf("AddTask error = %v", err)
	}

	tasks, err = LoadTasks()
	if err != nil || len(tasks) != 4 {
		t.Fatalf("expected 4 tasks after AddTask, got %d", len(tasks))
	}

	// Complete task 1
	if err := CompleteTask(1); err != nil {
		t.Fatalf("CompleteTask error = %v", err)
	}

	reminder := FormatTasksReminder()
	if reminder == "" {
		t.Fatal("expected non-empty task reminder string")
	}

	tasks, _ = LoadTasks()
	if !tasks[0].Completed {
		t.Errorf("expected task 1 to be completed")
	}
	if tasks[1].Completed {
		t.Errorf("expected task 2 to remain uncompleted")
	}

	// Clear tasks
	if err := ClearTasks(); err != nil {
		t.Fatalf("ClearTasks error = %v", err)
	}

	tasksAfterClear, err := LoadTasks()
	if err != nil {
		t.Fatalf("LoadTasks after clear error = %v", err)
	}
	if len(tasksAfterClear) != 0 {
		t.Errorf("expected 0 tasks after clear, got %d", len(tasksAfterClear))
	}
}
