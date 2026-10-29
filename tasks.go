package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

type Task struct {
	ID          int    `json:"id"`
	Description string `json:"description"`
	Completed   bool   `json:"completed"`
}

type TaskList struct {
	Tasks []Task `json:"tasks"`
}

var taskMu sync.Mutex

func GetTaskFilePath() string {
	if custom := os.Getenv("SCOOTER_TASK_PATH"); custom != "" {
		return custom
	}
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		cacheDir = ".cache"
	}
	dir := filepath.Join(cacheDir, "scooter")
	_ = os.MkdirAll(dir, 0755)
	return filepath.Join(dir, "tasks.json")
}

func LoadTasks() ([]Task, error) {
	taskMu.Lock()
	defer taskMu.Unlock()

	path := GetTaskFilePath()
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var tl TaskList
	if err := json.Unmarshal(data, &tl); err != nil {
		return nil, err
	}
	return tl.Tasks, nil
}

func SaveTasks(tasks []Task) error {
	taskMu.Lock()
	defer taskMu.Unlock()

	path := GetTaskFilePath()
	tl := TaskList{Tasks: tasks}
	data, err := json.MarshalIndent(tl, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func SetTasks(descriptions []string) error {
	var tasks []Task
	for i, desc := range descriptions {
		tasks = append(tasks, Task{
			ID:          i + 1,
			Description: desc,
			Completed:   false,
		})
	}
	return SaveTasks(tasks)
}

func AddTask(description string) error {
	tasks, _ := LoadTasks()
	id := 1
	if len(tasks) > 0 {
		id = tasks[len(tasks)-1].ID + 1
	}
	tasks = append(tasks, Task{
		ID:          id,
		Description: description,
		Completed:   false,
	})
	return SaveTasks(tasks)
}

func CompleteTask(id int) error {
	tasks, err := LoadTasks()
	if err != nil || len(tasks) == 0 {
		return err
	}
	for i := range tasks {
		if tasks[i].ID == id {
			tasks[i].Completed = true
			break
		}
	}
	return SaveTasks(tasks)
}

func ClearTasks() error {
	taskMu.Lock()
	defer taskMu.Unlock()
	path := GetTaskFilePath()
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func FormatTasksReminder() string {
	tasks, err := LoadTasks()
	if err != nil || len(tasks) == 0 {
		return ""
	}
	var uncompleted []string
	completedCount := 0
	for _, t := range tasks {
		if t.Completed {
			completedCount++
		} else {
			uncompleted = append(uncompleted, fmt.Sprintf("- [ ] [%d] %s", t.ID, t.Description))
		}
	}
	if len(uncompleted) == 0 {
		return "Current Task Status: All tasks completed successfully!"
	}
	return fmt.Sprintf("Current Task Progress (%d completed):\nPending steps:\n%s", completedCount, strings.Join(uncompleted, "\n"))
}
