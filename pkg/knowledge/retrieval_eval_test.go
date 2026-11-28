package knowledge

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

type fixedEmbedder struct {
	model   string
	version string
	vectors map[string][]float32
	err     error
}

func (e fixedEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	if e.err != nil {
		return nil, e.err
	}
	vector, ok := e.vectors[text]
	if !ok {
		return nil, errors.New("no fixed vector for input")
	}
	return vector, nil
}

func (e fixedEmbedder) Model() string        { return e.model }
func (e fixedEmbedder) ModelVersion() string { return e.version }

func openRetrievalTestDB(t *testing.T) *KnowledgeDB {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "knowledge.sqlite")
	t.Setenv("SCOOTER_DB_PATH", dbPath)
	kdb, err := Open()
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	t.Cleanup(func() {
		if err := kdb.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
	return kdb
}

// This small fixed-vector evaluation set checks semantic retrieval wiring
// without depending on the RKNN runtime or hardware quality characteristics.
func TestSearchWithEmbedderRetrievalEvaluation(t *testing.T) {
	type evalCase struct {
		query         string
		queryVector   []float32
		relevantTitle string
	}
	evaluation := []evalCase{
		{query: "How do I fix a vehicle malfunction?", queryVector: []float32{1, 0, 0}, relevantTitle: "Engine Repair"},
		{query: "Can I reduce premiums after a crash?", queryVector: []float32{0, 1, 0}, relevantTitle: "Auto Insurance"},
		{query: "How do I patch a bicycle tube?", queryVector: []float32{0, 0.9, 0.1}, relevantTitle: "Bicycle Tires"},
	}
	documents := []struct {
		title      string
		content    string
		collection string
		vector     []float32
	}{
		{title: "Engine Repair", content: "Mechanics inspect engines and service passenger vehicles.", collection: "transport", vector: []float32{1, 0, 0}},
		{title: "Auto Insurance", content: "Policies cover claims, premiums, and collision damage.", collection: "transport", vector: []float32{0, 1, 0}},
		{title: "Bicycle Tires", content: "Cycling tubes can be patched or replaced after a puncture.", collection: "transport", vector: []float32{0, 0.9, 0.1}},
		{title: "Bread Baking", content: "Mix flour, water, and yeast before proofing the dough.", collection: "cooking", vector: []float32{0, 0, 1}},
	}

	kdb := openRetrievalTestDB(t)
	embedder := fixedEmbedder{model: "test-model", version: "test-revision", vectors: make(map[string][]float32)}
	for _, doc := range documents {
		embedder.vectors[doc.content] = doc.vector
		if _, err := kdb.IngestTextWithEmbedder(context.Background(), doc.title, doc.content, doc.collection, doc.title, embedder); err != nil {
			t.Fatalf("ingest %q: %v", doc.title, err)
		}
	}
	for _, testCase := range evaluation {
		embedder.vectors[testCase.query] = testCase.queryVector
	}

	for _, testCase := range evaluation {
		t.Run(testCase.query, func(t *testing.T) {
			results, err := kdb.SearchWithEmbedder(context.Background(), testCase.query, "transport", 3, embedder)
			if err != nil {
				t.Fatalf("SearchWithEmbedder() error = %v", err)
			}
			if len(results) == 0 {
				t.Fatal("expected at least one result")
			}
			if results[0].Title != testCase.relevantTitle {
				t.Errorf("top result = %q, want %q", results[0].Title, testCase.relevantTitle)
			}
		})
	}
}

func TestSearchWithEmbedderFiltersCollectionAndModelVersion(t *testing.T) {
	kdb := openRetrievalTestDB(t)
	old := fixedEmbedder{model: "test-model", version: "v1", vectors: map[string][]float32{"old collection match": {1, 0}, "query": {1, 0}}}
	current := fixedEmbedder{model: "test-model", version: "v2", vectors: map[string][]float32{"current collection match": {1, 0}, "other collection match": {1, 0}, "query": {1, 0}}}
	for _, item := range []struct {
		title      string
		content    string
		collection string
		embedder   Embedder
	}{
		{title: "Old Version", content: "old collection match", collection: "alpha", embedder: old},
		{title: "Current Alpha", content: "current collection match", collection: "alpha", embedder: current},
		{title: "Current Beta", content: "other collection match", collection: "beta", embedder: current},
	} {
		if _, err := kdb.IngestTextWithEmbedder(context.Background(), item.title, item.content, item.collection, item.title, item.embedder); err != nil {
			t.Fatalf("ingest %q: %v", item.title, err)
		}
	}

	results, err := kdb.SearchWithEmbedder(context.Background(), "query", "alpha", 5, current)
	if err != nil {
		t.Fatalf("SearchWithEmbedder() error = %v", err)
	}
	if len(results) != 1 || results[0].Title != "Current Alpha" {
		t.Fatalf("results = %#v, want only Current Alpha", results)
	}
}

func TestReciprocalRankFusionCombinesRankings(t *testing.T) {
	lexical := []SearchResult{{ChunkID: "a"}, {ChunkID: "b"}}
	dense := []SearchResult{{ChunkID: "b"}, {ChunkID: "c"}}
	results := reciprocalRankFusion(lexical, dense, 3)
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}
	if results[0].ChunkID != "b" {
		t.Errorf("top fused result = %q, want shared result b", results[0].ChunkID)
	}
}

func TestSearchWithEmbedderFallsBackToLexicalWhenNil(t *testing.T) {
	kdb := openRetrievalTestDB(t)
	if _, err := kdb.IngestText("Lexical", "The distinctive word quokka appears here.", "alpha", "lexical.md"); err != nil {
		t.Fatalf("IngestText() error = %v", err)
	}
	results, err := kdb.SearchWithEmbedder(context.Background(), "quokka", "", 5, nil)
	if err != nil {
		t.Fatalf("SearchWithEmbedder() error = %v", err)
	}
	if len(results) != 1 || results[0].Title != "Lexical" {
		t.Fatalf("results = %#v, want lexical result", results)
	}
}

func TestIngestFileUsesCanonicalSourceForReplacement(t *testing.T) {
	kdb := openRetrievalTestDB(t)
	path := filepath.Join(t.TempDir(), "source.txt")
	if err := os.WriteFile(path, []byte("first version content"), 0600); err != nil {
		t.Fatal(err)
	}
	firstID, err := kdb.IngestFile(path, "alpha")
	if err != nil {
		t.Fatalf("first IngestFile() error = %v", err)
	}
	if err := os.WriteFile(path, []byte("updated version content"), 0600); err != nil {
		t.Fatal(err)
	}
	secondID, err := kdb.IngestFile(filepath.Join(filepath.Dir(path), ".", filepath.Base(path)), "alpha")
	if err != nil {
		t.Fatalf("second IngestFile() error = %v", err)
	}
	if firstID == secondID {
		t.Fatal("replacement should create a new document ID")
	}
	oldResults, err := kdb.Search("first", "alpha", 5)
	if err != nil {
		t.Fatalf("Search(old content) error = %v", err)
	}
	if len(oldResults) != 0 {
		t.Errorf("old full-text entries remain after replacement: %#v", oldResults)
	}
	newResults, err := kdb.Search("updated", "alpha", 5)
	if err != nil {
		t.Fatalf("Search(new content) error = %v", err)
	}
	if len(newResults) != 1 || newResults[0].DocumentID != secondID {
		t.Errorf("new content results = %#v, want replacement document %q", newResults, secondID)
	}
}
