package main

import (
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"codeberg.org/indievisible/scooter/pkg/knowledge"
	"codeberg.org/indievisible/scooter/pkg/rknn"
)

var (
	knowledgeEmbedderOnce sync.Once
	knowledgeEmbedder     knowledge.Embedder
	knowledgeEmbedderErr  error
)

// getKnowledgeEmbedder enables local embeddings when a model directory is
// configured or the standard BGE-M3 model directory exists. With no model
// installed, ingestion continues with lexical indexing only.
func getKnowledgeEmbedder() (knowledge.Embedder, error) {
	knowledgeEmbedderOnce.Do(func() {
		modelDir := strings.TrimSpace(os.Getenv("SCOOTER_BGE_M3_MODEL_DIR"))
		if modelDir == "" {
			home, err := os.UserHomeDir()
			if err != nil {
				knowledgeEmbedderErr = err
				return
			}
			modelDir = filepath.Join(home, "3588-stuff", "rknn-models", "bge-m3-rknn2-rk3588")
			if _, err := os.Stat(filepath.Join(modelDir, "bge-m3_dynamic.rknn")); err != nil {
				log.Printf("BGE-M3 model not found at %s; using lexical-only knowledge search", modelDir)
				return
			}
		}
		knowledgeEmbedder, knowledgeEmbedderErr = rknn.NewBGEEmbedder(modelDir)
	})
	return knowledgeEmbedder, knowledgeEmbedderErr
}
