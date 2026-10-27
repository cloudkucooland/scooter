package knowledge

import (
	"os"
	"path/filepath"
	"testing"
)

func TestKnowledgeIngestAndSearch(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "scooter_knowledge_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	dbPath := filepath.Join(tmpDir, "test_knowledge.sqlite")
	t.Setenv("SCOOTER_DB_PATH", dbPath)

	kdb, err := Open()
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer kdb.Close()

	docID, err := kdb.IngestText("Grace in Romans", "Where sin increased, grace abounded all the more. This is a powerful illustration of unmerited favor.", "sermons-2026", "romans.md")
	if err != nil {
		t.Fatalf("IngestText() error = %v", err)
	}
	if docID == "" {
		t.Fatal("expected non-empty document ID")
	}

	results, err := kdb.Search("grace", "", 5)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected search results for 'grace', got 0")
	}
	if results[0].Title != "Grace in Romans" {
		t.Errorf("expected title 'Grace in Romans', got %q", results[0].Title)
	}
}

func TestKnowledgeVectorSearchAndReembed(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "scooter_vector_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	dbPath := filepath.Join(tmpDir, "test_vector.sqlite")
	t.Setenv("SCOOTER_DB_PATH", dbPath)

	kdb, err := Open()
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer kdb.Close()

	_, err = kdb.IngestText("Sermon on Grace", "Grace is amazing.", "sermons", "grace.md")
	if err != nil {
		t.Fatalf("IngestText() error = %v", err)
	}

	mockEmbed := func(content string) ([]float32, error) {
		return []float32{1.0, 0.0, 0.0, 0.0}, nil
	}

	err = kdb.ReembedChunks("qwen3.8-embedding", mockEmbed)
	if err != nil {
		t.Fatalf("ReembedChunks() error = %v", err)
	}

	results, err := kdb.VectorSearch([]float32{1.0, 0.0, 0.0, 0.0}, "qwen3.8-embedding", "", 5)
	if err != nil {
		t.Fatalf("VectorSearch() error = %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected vector search results, got 0")
	}
	if results[0].Score < 0.99 {
		t.Errorf("expected high similarity score, got %f", results[0].Score)
	}
}
