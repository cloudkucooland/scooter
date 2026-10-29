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

func TestSearchHybridWithoutVectorModel(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "scooter_hybrid_no_vec")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	t.Setenv("SCOOTER_DB_PATH", filepath.Join(tmpDir, "hybrid.sqlite"))
	kdb, err := Open()
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer kdb.Close()

	_, err = kdb.IngestText("Test Title", "Golang hybrid search testing without vector model.", "col1", "test.md")
	if err != nil {
		t.Fatalf("IngestText error = %v", err)
	}

	results, err := kdb.SearchHybrid("golang", "col1", 5)
	if err != nil {
		t.Fatalf("SearchHybrid error = %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected hybrid search results, got 0")
	}
}

func TestSearchHybridWithVectorModel(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "scooter_hybrid_vec")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	t.Setenv("SCOOTER_DB_PATH", filepath.Join(tmpDir, "hybrid_vec.sqlite"))
	kdb, err := Open()
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer kdb.Close()

	kdb.EmbedFunc = func(s string) ([]float32, error) {
		return []float32{1.0, 0.0, 0.0}, nil
	}
	kdb.QueryEmbedFunc = func(s string) ([]float32, error) {
		return []float32{1.0, 0.0, 0.0}, nil
	}

	_, err = kdb.IngestText("Vector Test", "Semantic vector search content.", "col-vec", "vec.md")
	if err != nil {
		t.Fatalf("IngestText error = %v", err)
	}

	err = kdb.ReembedChunks("test-model", kdb.EmbedFunc)
	if err != nil {
		t.Fatalf("ReembedChunks error = %v", err)
	}

	results, err := kdb.SearchHybrid("semantic", "col-vec", 5)
	if err != nil {
		t.Fatalf("SearchHybrid error = %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected hybrid search results with vector model, got 0")
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

	// VectorSearch with non-existent model
	vRes, err := kdb.VectorSearch([]float32{1.0, 0.0}, "bad-model", "", 5)
	if err != nil {
		t.Errorf("VectorSearch error = %v", err)
	}
	if len(vRes) != 0 {
		t.Errorf("expected 0 results for bad model, got %d", len(vRes))
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

	// Insert test nodes and edges via raw SQL
	_, err = kdb.db.Exec(`INSERT INTO nodes (id, label, properties) VALUES (?, ?, ?)`, "node_1", "Person", "{}")
	if err != nil {
		t.Fatalf("insert node 1: %v", err)
	}
	_, err = kdb.db.Exec(`INSERT INTO nodes (id, label, properties) VALUES (?, ?, ?)`, "node_2", "Concept", "{}")
	if err != nil {
		t.Fatalf("insert node 2: %v", err)
	}
	_, err = kdb.db.Exec(`INSERT INTO edges (source_id, target_id, relation, weight) VALUES (?, ?, ?, ?)`, "node_1", "node_2", "knows", 1.0)
	if err != nil {
		t.Fatalf("insert edge: %v", err)
	}

	edges, err := kdb.QueryGraph("node_1", "knows")
	if err != nil {
		t.Fatalf("QueryGraph error = %v", err)
	}
	if len(edges) != 1 {
		t.Fatalf("expected 1 edge, got %d", len(edges))
	}
	if edges[0].TargetID != "node_2" || edges[0].Relation != "knows" {
		t.Errorf("unexpected edge result: %+v", edges[0])
	}

	// Query graph without relation filter
	edgesAll, err := kdb.QueryGraph("node_1", "")
	if err != nil {
		t.Fatalf("QueryGraph (no relation) error = %v", err)
	}
	if len(edgesAll) != 1 {
		t.Errorf("expected 1 edge without relation filter, got %d", len(edgesAll))
	}
}

func TestAddNodeAndEdge(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "scooter_add_graph")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	t.Setenv("SCOOTER_DB_PATH", filepath.Join(tmpDir, "add_graph.sqlite"))
	kdb, err := Open()
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer kdb.Close()

	err = kdb.AddNode("sermon_1", "Sermon", `{"title":"Grace"}`)
	if err != nil {
		t.Fatalf("AddNode sermon_1 error = %v", err)
	}

	err = kdb.AddNode("metaphor_anchor", "Metaphor", `{"name":"Anchor"}`)
	if err != nil {
		t.Fatalf("AddNode metaphor_anchor error = %v", err)
	}

	err = kdb.AddEdge("sermon_1", "metaphor_anchor", "USES_METAPHOR", 1.5)
	if err != nil {
		t.Fatalf("AddEdge error = %v", err)
	}

	edges, err := kdb.QueryGraph("sermon_1", "USES_METAPHOR")
	if err != nil {
		t.Fatalf("QueryGraph error = %v", err)
	}
	if len(edges) != 1 {
		t.Fatalf("expected 1 edge, got %d", len(edges))
	}
	if edges[0].TargetID != "metaphor_anchor" || edges[0].Relation != "USES_METAPHOR" || edges[0].Weight != 1.5 {
		t.Errorf("unexpected edge values: %+v", edges[0])
	}
}
