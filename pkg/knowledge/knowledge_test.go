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

func TestSplitIntoChunksUTF8(t *testing.T) {
	text := "你好世界 🌍\n\n测试多字节字符的中文分块是否正确无误。"
	chunks := splitIntoChunks(text, 10)
	if len(chunks) == 0 {
		t.Fatal("expected chunks, got 0")
	}
	for _, c := range chunks {
		if c == "" {
			t.Error("expected non-empty chunk")
		}
	}
}

func TestOversizedParagraphChunking(t *testing.T) {
	// Oversized paragraph without double newlines, containing UTF-8 characters
	longText := "这是一个非常长的段落，包含中文字符和英文 English characters 1234567890，用来测试当段落超过 maxChars 时是否能够正确按 rune 边界切分而不乱码。"
	chunks := splitIntoChunks(longText, 15)
	if len(chunks) <= 1 {
		t.Errorf("expected multiple chunks for oversized paragraph, got %d", len(chunks))
	}
	for _, c := range chunks {
		if len([]rune(c)) > 15 {
			t.Errorf("chunk exceeds maxChars (15 runes): %q (len %d)", c, len([]rune(c)))
		}
	}
}

func TestSearchEdgeCases(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "scooter_edge")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	t.Setenv("SCOOTER_DB_PATH", filepath.Join(tmpDir, "edge.sqlite"))
	kdb, err := Open()
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer kdb.Close()

	_, err = kdb.IngestText("Edge Doc", "Content for edge testing.", "collection-a", "edge.md")
	if err != nil {
		t.Fatalf("IngestText error = %v", err)
	}

	// Empty query / zero limit
	res, err := kdb.Search("", "", 0)
	if err != nil {
		t.Errorf("Search with empty query and zero limit error = %v", err)
	}
	_ = res

	// Non-existent collection search
	res2, err := kdb.Search("content", "non-existent-collection", 5)
	if err != nil {
		t.Errorf("Search with missing collection error = %v", err)
	}
	if len(res2) != 0 {
		t.Errorf("expected 0 results for missing collection, got %d", len(res2))
	}
}

func TestListCollections(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "scooter_cols")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	t.Setenv("SCOOTER_DB_PATH", filepath.Join(tmpDir, "cols.sqlite"))
	kdb, err := Open()
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer kdb.Close()

	_, _ = kdb.IngestText("Doc 1", "Content 1", "sermons", "s1.md")
	_, _ = kdb.IngestText("Doc 2", "Content 2", "articles", "a1.md")

	cols, err := kdb.ListCollections()
	if err != nil {
		t.Fatalf("ListCollections error = %v", err)
	}
	if len(cols) != 2 {
		t.Errorf("expected 2 collections, got %d (%v)", len(cols), cols)
	}
}

func TestQueryGraph(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "scooter_graph")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	t.Setenv("SCOOTER_DB_PATH", filepath.Join(tmpDir, "graph.sqlite"))
	kdb, err := Open()
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer kdb.Close()

	_ = kdb.AddNode("node1", "Topic", "{\"desc\":\"test\"}")
	_ = kdb.AddNode("node2", "Concept", "{\"desc\":\"test2\"}")
	_ = kdb.AddEdge("node1", "node2", "RELATED", 1.5)

	edges, err := kdb.QueryGraph("node1", "RELATED")
	if err != nil {
		t.Fatalf("QueryGraph error = %v", err)
	}
	if len(edges) != 1 {
		t.Errorf("expected 1 edge, got %d", len(edges))
	}
	if edges[0].TargetID != "node2" || edges[0].Weight != 1.5 {
		t.Errorf("unexpected edge details: %+v", edges[0])
	}
}

func TestAddNodeAndEdge(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "scooter_node_edge")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	t.Setenv("SCOOTER_DB_PATH", filepath.Join(tmpDir, "ne.sqlite"))
	kdb, err := Open()
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer kdb.Close()

	if err := kdb.AddNode("a", "LabelA", "prop"); err != nil {
		t.Fatalf("AddNode error = %v", err)
	}
	if err := kdb.AddEdge("a", "b", "CONNECTS", 2.0); err != nil {
		t.Fatalf("AddEdge error = %v", err)
	}
}
