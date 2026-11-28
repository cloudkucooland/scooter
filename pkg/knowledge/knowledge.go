package knowledge

import (
	"context"
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

type Document struct {
	ID         string    `json:"id"`
	Title      string    `json:"title"`
	SourcePath string    `json:"source_path"`
	Collection string    `json:"collection"`
	Content    string    `json:"content"`
	Metadata   string    `json:"metadata"`
	UpdatedAt  time.Time `json:"updated_at"`
}

type Chunk struct {
	ID         string `json:"id"`
	DocumentID string `json:"document_id"`
	ChunkIndex int    `json:"chunk_index"`
	Content    string `json:"content"`
}

// ChunkEmbedding identifies a vector by the model and model version that
// produced it. Multiple versions can coexist for the same chunk.
type ChunkEmbedding struct {
	ChunkID      string    `json:"chunk_id"`
	Model        string    `json:"model"`
	ModelVersion string    `json:"model_version"`
	Dimensions   int       `json:"dimensions"`
	Vector       []float32 `json:"vector"`
}

type SearchResult struct {
	ChunkID    string  `json:"chunk_id"`
	DocumentID string  `json:"document_id"`
	Title      string  `json:"title"`
	Collection string  `json:"collection"`
	Content    string  `json:"content"`
	Score      float64 `json:"score"`
}

type KnowledgeDB struct {
	db *sql.DB
}

// Embedder produces a vector and identifies the model revision that produced it.
type Embedder interface {
	Embed(context.Context, string) ([]float32, error)
	Model() string
	ModelVersion() string
}

func GetDBPath() string {
	if custom := os.Getenv("SCOOTER_DB_PATH"); custom != "" {
		return custom
	}
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		cacheDir = ".cache"
	}
	dir := filepath.Join(cacheDir, "scooter")
	_ = os.MkdirAll(dir, 0755)
	return filepath.Join(dir, "knowledge.sqlite")
}

func Open() (*KnowledgeDB, error) {
	dbPath := GetDBPath()
	if err := os.MkdirAll(filepath.Dir(dbPath), 0755); err != nil {
		return nil, fmt.Errorf("mkdir db dir: %w", err)
	}

	dsn := fmt.Sprintf("%s?_pragma=foreign_keys(1)&_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)", dbPath)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite db: %w", err)
	}

	kdb := &KnowledgeDB{db: db}
	if err := kdb.initSchema(); err != nil {
		db.Close()
		return nil, err
	}

	return kdb, nil
}

func (k *KnowledgeDB) Close() error {
	return k.db.Close()
}

func (k *KnowledgeDB) initSchema() error {
	schema := `
	CREATE TABLE IF NOT EXISTS documents (
		id TEXT PRIMARY KEY,
		title TEXT,
		source_path TEXT,
		collection TEXT,
		content TEXT,
		metadata TEXT,
		updated_at DATETIME
	);

	CREATE TABLE IF NOT EXISTS chunks (
		id TEXT PRIMARY KEY,
		document_id TEXT,
		chunk_index INTEGER,
		content TEXT,
		FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
	);

	CREATE TABLE IF NOT EXISTS chunk_embeddings (
		chunk_id TEXT NOT NULL,
		model TEXT NOT NULL,
		model_version TEXT NOT NULL,
		dimensions INTEGER NOT NULL CHECK (dimensions > 0),
		vector BLOB NOT NULL,
		created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
		PRIMARY KEY (chunk_id, model, model_version),
		FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
	);
	CREATE INDEX IF NOT EXISTS chunk_embeddings_model_version_idx
		ON chunk_embeddings(model, model_version, chunk_id);

	CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
		content,
		chunk_id UNINDEXED,
		document_id UNINDEXED
	);

	CREATE TABLE IF NOT EXISTS nodes (
		id TEXT PRIMARY KEY,
		label TEXT,
		properties TEXT
	);

	CREATE TABLE IF NOT EXISTS edges (
		source_id TEXT,
		target_id TEXT,
		relation TEXT,
		weight REAL,
		PRIMARY KEY(source_id, target_id, relation)
	);
	`
	if _, err := k.db.Exec(schema); err != nil {
		return err
	}
	return nil
}

// StoreChunkEmbedding inserts or replaces a vector for the given chunk and
// embedding model version. Vectors are stored as little-endian float32 BLOBs.
func (k *KnowledgeDB) StoreChunkEmbedding(chunkID, model, modelVersion string, vector []float32) error {
	return storeChunkEmbedding(k.db, chunkID, model, modelVersion, vector)
}

type embeddingExecer interface {
	Exec(query string, args ...any) (sql.Result, error)
}

func storeChunkEmbedding(exec embeddingExecer, chunkID, model, modelVersion string, vector []float32) error {
	if strings.TrimSpace(chunkID) == "" || strings.TrimSpace(model) == "" || strings.TrimSpace(modelVersion) == "" {
		return fmt.Errorf("chunk ID, model, and model version are required")
	}
	if len(vector) == 0 {
		return fmt.Errorf("embedding vector must not be empty")
	}
	blob := make([]byte, len(vector)*4)
	for i, value := range vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("embedding vector contains non-finite value at index %d", i)
		}
		binary.LittleEndian.PutUint32(blob[i*4:], math.Float32bits(value))
	}
	_, err := exec.Exec(`
		INSERT INTO chunk_embeddings (chunk_id, model, model_version, dimensions, vector)
		VALUES (?, ?, ?, ?, ?)
		ON CONFLICT(chunk_id, model, model_version) DO UPDATE SET
			dimensions = excluded.dimensions, vector = excluded.vector, created_at = CURRENT_TIMESTAMP
	`, chunkID, model, modelVersion, len(vector), blob)
	return err
}

// GetChunkEmbedding fetches an exact model/version vector for a chunk.
func (k *KnowledgeDB) GetChunkEmbedding(chunkID, model, modelVersion string) (ChunkEmbedding, error) {
	var embedding ChunkEmbedding
	var blob []byte
	err := k.db.QueryRow(`
		SELECT chunk_id, model, model_version, dimensions, vector
		FROM chunk_embeddings WHERE chunk_id = ? AND model = ? AND model_version = ?
	`, chunkID, model, modelVersion).Scan(&embedding.ChunkID, &embedding.Model, &embedding.ModelVersion, &embedding.Dimensions, &blob)
	if err != nil {
		return ChunkEmbedding{}, err
	}
	if embedding.Dimensions <= 0 || len(blob) != embedding.Dimensions*4 {
		return ChunkEmbedding{}, fmt.Errorf("invalid stored embedding dimensions %d for %d-byte vector", embedding.Dimensions, len(blob))
	}
	embedding.Vector = make([]float32, embedding.Dimensions)
	for i := range embedding.Vector {
		embedding.Vector[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
	}
	return embedding, nil
}

func splitIntoChunks(content string, maxChars int) []string {
	if maxChars <= 0 {
		maxChars = 1000
	}
	paragraphs := strings.Split(content, "\n\n")
	var chunks []string
	var current strings.Builder

	for _, p := range paragraphs {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if current.Len()+len(p)+2 > maxChars {
			if current.Len() > 0 {
				chunks = append(chunks, current.String())
				current.Reset()
			}
			if len([]rune(p)) > maxChars {
				runes := []rune(p)
				for i := 0; i < len(runes); i += maxChars {
					end := i + maxChars
					if end > len(runes) {
						end = len(runes)
					}
					chunks = append(chunks, string(runes[i:end]))
				}
				continue
			}
		}
		if current.Len() > 0 {
			current.WriteString("\n\n")
		}
		current.WriteString(p)
	}

	if current.Len() > 0 {
		chunks = append(chunks, current.String())
	}

	if len(chunks) == 0 && strings.TrimSpace(content) != "" {
		chunks = []string{content}
	}

	return chunks
}

func (k *KnowledgeDB) ListCollections() ([]string, error) {
	rows, err := k.db.Query(`SELECT DISTINCT collection FROM documents WHERE collection != '' ORDER BY collection`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var cols []string
	for rows.Next() {
		var c string
		if err := rows.Scan(&c); err != nil {
			return nil, err
		}
		cols = append(cols, c)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return cols, nil
}

func (k *KnowledgeDB) IngestText(title, content, collection, sourcePath string) (string, error) {
	return k.IngestTextWithEmbedder(context.Background(), title, content, collection, sourcePath, nil)
}

// IngestTextWithEmbedder stores chunks and, when configured, their embeddings.
// It computes vectors before opening the write transaction so NPU inference does
// not hold SQLite's write lock; chunk and vector rows then commit atomically.
func (k *KnowledgeDB) IngestTextWithEmbedder(ctx context.Context, title, content, collection, sourcePath string, embedder Embedder) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	chunks := splitIntoChunks(content, 1000)
	embeddings := make([][]float32, len(chunks))
	if embedder != nil {
		for i, chunk := range chunks {
			vector, err := embedder.Embed(ctx, chunk)
			if err != nil {
				return "", fmt.Errorf("embed chunk %d: %w", i, err)
			}
			embeddings[i] = vector
		}
	}
	return k.ingestChunks(title, content, collection, sourcePath, chunks, embeddings, embedder)
}

func (k *KnowledgeDB) ingestChunks(title, content, collection, sourcePath string, chunks []string, embeddings [][]float32, embedder Embedder) (string, error) {
	tx, err := k.db.Begin()
	if err != nil {
		return "", err
	}
	defer tx.Rollback()

	// FTS5 rows do not have a foreign key to documents, so remove them before
	// deleting existing documents. Chunk embeddings are removed by the chunks'
	// ON DELETE CASCADE foreign keys.
	if strings.TrimSpace(sourcePath) != "" {
		if _, err := tx.Exec(`
			DELETE FROM chunks_fts
			WHERE document_id IN (SELECT id FROM documents WHERE source_path = ?)
		`, sourcePath); err != nil {
			return "", fmt.Errorf("remove old full-text entries: %w", err)
		}
		if _, err := tx.Exec(`DELETE FROM documents WHERE source_path = ?`, sourcePath); err != nil {
			return "", fmt.Errorf("replace existing source document: %w", err)
		}
	} else {
		// Text-only ingestion has no stable path. Treat title within a collection
		// as its identity so re-ingestion also removes stale chunks and vectors.
		if _, err := tx.Exec(`
			DELETE FROM chunks_fts
			WHERE document_id IN (SELECT id FROM documents WHERE source_path = '' AND collection = ? AND title = ?)
		`, collection, title); err != nil {
			return "", fmt.Errorf("remove old full-text entries: %w", err)
		}
		if _, err := tx.Exec(`DELETE FROM documents WHERE source_path = '' AND collection = ? AND title = ?`, collection, title); err != nil {
			return "", fmt.Errorf("replace existing text document: %w", err)
		}
	}

	docID := fmt.Sprintf("doc_%d", time.Now().UnixNano())
	now := time.Now().Format(time.RFC3339)

	_, err = tx.Exec(
		`INSERT INTO documents (id, title, source_path, collection, content, metadata, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)`,
		docID, title, sourcePath, collection, content, "{}", now,
	)
	if err != nil {
		return "", err
	}

	for i, chunkContent := range chunks {
		chunkID := fmt.Sprintf("%s_chunk_%d", docID, i)
		_, err = tx.Exec(
			`INSERT INTO chunks (id, document_id, chunk_index, content) VALUES (?, ?, ?, ?)`,
			chunkID, docID, i, chunkContent,
		)
		if err != nil {
			return "", err
		}

		_, err = tx.Exec(
			`INSERT INTO chunks_fts (content, chunk_id, document_id) VALUES (?, ?, ?)`,
			chunkContent, chunkID, docID,
		)
		if err != nil {
			return "", err
		}
		if embedder != nil {
			if err := storeChunkEmbedding(tx, chunkID, embedder.Model(), embedder.ModelVersion(), embeddings[i]); err != nil {
				return "", fmt.Errorf("store embedding for chunk %d: %w", i, err)
			}
		}
	}

	return docID, tx.Commit()
}

func (k *KnowledgeDB) IngestFile(filePath, collection string) (string, error) {
	return k.IngestFileWithEmbedder(context.Background(), filePath, collection, nil)
}

// IngestFileWithEmbedder reads a file and stores chunk vectors when embedder is non-nil.
func (k *KnowledgeDB) IngestFileWithEmbedder(ctx context.Context, filePath, collection string, embedder Embedder) (string, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}
	sourcePath, err := filepath.Abs(filePath)
	if err != nil {
		return "", fmt.Errorf("resolve ingestion source path: %w", err)
	}
	sourcePath = filepath.Clean(sourcePath)
	title := filepath.Base(filePath)
	return k.IngestTextWithEmbedder(ctx, title, string(data), collection, sourcePath, embedder)
}

func (k *KnowledgeDB) Search(query string, collection string, limit int) ([]SearchResult, error) {
	if limit <= 0 {
		limit = 5
	}

	var rows *sql.Rows
	var err error
	ftsQuery := quoteFTSQuery(query)
	if collection != "" {
		rows, err = k.db.Query(`
			SELECT c.id, d.id, d.title, d.collection, c.content, f.rank
			FROM chunks_fts f
			JOIN chunks c ON c.id = f.chunk_id
			JOIN documents d ON d.id = c.document_id
			WHERE chunks_fts MATCH ? AND d.collection = ?
			ORDER BY f.rank
			LIMIT ?
		`, ftsQuery, collection, limit)
	} else {
		rows, err = k.db.Query(`
			SELECT c.id, d.id, d.title, d.collection, c.content, f.rank
			FROM chunks_fts f
			JOIN chunks c ON c.id = f.chunk_id
			JOIN documents d ON d.id = c.document_id
			WHERE chunks_fts MATCH ?
			ORDER BY f.rank
			LIMIT ?
		`, ftsQuery, limit)
	}
	if err != nil {
		return k.searchLike(query, collection, limit)
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var sr SearchResult
		if err := rows.Scan(&sr.ChunkID, &sr.DocumentID, &sr.Title, &sr.Collection, &sr.Content, &sr.Score); err != nil {
			return nil, err
		}
		results = append(results, sr)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return k.searchLike(query, collection, limit)
	}
	return results, nil
}

// quoteFTSQuery turns user text into a literal phrase, avoiding FTS5 operator
// and punctuation parsing errors while retaining ordinary lexical matching.
func quoteFTSQuery(query string) string {
	return `"` + strings.ReplaceAll(strings.TrimSpace(query), `"`, `""`) + `"`
}

// SearchWithEmbedder combines lexical and vector rankings with reciprocal rank
// fusion. Vector candidates are restricted to the embedder's exact model
// version so incompatible dimensions or revisions are never mixed.
func (k *KnowledgeDB) SearchWithEmbedder(ctx context.Context, query, collection string, limit int, embedder Embedder) ([]SearchResult, error) {
	if limit <= 0 {
		limit = 5
	}
	if embedder == nil {
		return k.Search(query, collection, limit)
	}
	if ctx == nil {
		ctx = context.Background()
	}
	queryVector, err := embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed knowledge search query: %w", err)
	}
	candidateLimit := limit * 4
	if candidateLimit < 20 {
		candidateLimit = 20
	}
	if candidateLimit > 200 {
		candidateLimit = 200
	}
	lexical, err := k.Search(query, collection, candidateLimit)
	if err != nil {
		return nil, err
	}
	dense, err := k.searchVectors(queryVector, collection, embedder.Model(), embedder.ModelVersion(), candidateLimit)
	if err != nil {
		return nil, err
	}
	return reciprocalRankFusion(lexical, dense, limit), nil
}

func (k *KnowledgeDB) searchVectors(queryVector []float32, collection, model, modelVersion string, limit int) ([]SearchResult, error) {
	if len(queryVector) == 0 {
		return nil, fmt.Errorf("query embedding must not be empty")
	}
	queryNorm := vectorNorm(queryVector)
	if queryNorm == 0 || math.IsNaN(queryNorm) || math.IsInf(queryNorm, 0) {
		return nil, fmt.Errorf("query embedding has invalid norm")
	}
	query := `
		SELECT c.id, d.id, d.title, d.collection, c.content, e.dimensions, e.vector
		FROM chunk_embeddings e
		JOIN chunks c ON c.id = e.chunk_id
		JOIN documents d ON d.id = c.document_id
		WHERE e.model = ? AND e.model_version = ?`
	args := []any{model, modelVersion}
	if collection != "" {
		query += ` AND d.collection = ?`
		args = append(args, collection)
	}
	rows, err := k.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	results := make([]SearchResult, 0)
	for rows.Next() {
		var result SearchResult
		var dimensions int
		var blob []byte
		if err := rows.Scan(&result.ChunkID, &result.DocumentID, &result.Title, &result.Collection, &result.Content, &dimensions, &blob); err != nil {
			return nil, err
		}
		if dimensions != len(queryVector) || len(blob) != dimensions*4 {
			return nil, fmt.Errorf("invalid stored embedding for chunk %s: dimensions %d, bytes %d, query dimensions %d", result.ChunkID, dimensions, len(blob), len(queryVector))
		}
		var dot, storedNormSquared float64
		for i, queryValue := range queryVector {
			storedValue := math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
			dot += float64(queryValue) * float64(storedValue)
			storedNormSquared += float64(storedValue) * float64(storedValue)
		}
		if storedNormSquared == 0 || math.IsNaN(storedNormSquared) || math.IsInf(storedNormSquared, 0) {
			return nil, fmt.Errorf("stored embedding for chunk %s has invalid norm", result.ChunkID)
		}
		result.Score = dot / (queryNorm * math.Sqrt(storedNormSquared))
		results = append(results, result)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].Score == results[j].Score {
			return results[i].ChunkID < results[j].ChunkID
		}
		return results[i].Score > results[j].Score
	})
	if len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

func vectorNorm(vector []float32) float64 {
	var normSquared float64
	for _, value := range vector {
		normSquared += float64(value) * float64(value)
	}
	return math.Sqrt(normSquared)
}

func reciprocalRankFusion(lexical, dense []SearchResult, limit int) []SearchResult {
	const rankConstant = 60.0
	type fusedResult struct {
		result SearchResult
		score  float64
	}
	fused := make(map[string]*fusedResult, len(lexical)+len(dense))
	addRanked := func(results []SearchResult) {
		for rank, result := range results {
			entry, ok := fused[result.ChunkID]
			if !ok {
				entry = &fusedResult{result: result}
				fused[result.ChunkID] = entry
			}
			entry.score += 1 / (rankConstant + float64(rank+1))
		}
	}
	addRanked(lexical)
	addRanked(dense)
	results := make([]fusedResult, 0, len(fused))
	for _, entry := range fused {
		entry.result.Score = entry.score
		results = append(results, *entry)
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].score == results[j].score {
			return results[i].result.ChunkID < results[j].result.ChunkID
		}
		return results[i].score > results[j].score
	})
	if len(results) > limit {
		results = results[:limit]
	}
	merged := make([]SearchResult, len(results))
	for i, result := range results {
		merged[i] = result.result
	}
	return merged
}

func (k *KnowledgeDB) searchLike(query string, collection string, limit int) ([]SearchResult, error) {
	var rows *sql.Rows
	var err error
	if collection != "" {
		rows, err = k.db.Query(`
			SELECT c.id, d.id, d.title, d.collection, c.content
			FROM chunks c
			JOIN documents d ON d.id = c.document_id
			WHERE c.content LIKE ? AND d.collection = ?
			LIMIT ?
		`, "%"+query+"%", collection, limit)
	} else {
		rows, err = k.db.Query(`
			SELECT c.id, d.id, d.title, d.collection, c.content
			FROM chunks c
			JOIN documents d ON d.id = c.document_id
			WHERE c.content LIKE ?
			LIMIT ?
		`, "%"+query+"%", limit)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var sr SearchResult
		if err := rows.Scan(&sr.ChunkID, &sr.DocumentID, &sr.Title, &sr.Collection, &sr.Content); err != nil {
			return nil, err
		}
		sr.Score = 1.0
		results = append(results, sr)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return results, nil
}

type EdgeResult struct {
	SourceID string  `json:"source_id"`
	TargetID string  `json:"target_id"`
	Relation string  `json:"relation"`
	Weight   float64 `json:"weight"`
}

func (k *KnowledgeDB) QueryGraph(sourceID, relation string) ([]EdgeResult, error) {
	var rows *sql.Rows
	var err error
	if relation != "" {
		rows, err = k.db.Query(`
			SELECT source_id, target_id, relation, weight
			FROM edges
			WHERE source_id = ? AND relation = ?
		`, sourceID, relation)
	} else {
		rows, err = k.db.Query(`
			SELECT source_id, target_id, relation, weight
			FROM edges
			WHERE source_id = ?
		`, sourceID)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var edges []EdgeResult
	for rows.Next() {
		var e EdgeResult
		if err := rows.Scan(&e.SourceID, &e.TargetID, &e.Relation, &e.Weight); err != nil {
			return nil, err
		}
		edges = append(edges, e)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return edges, nil
}

func (k *KnowledgeDB) AddNode(id, label, properties string) error {
	_, err := k.db.Exec(
		`INSERT INTO nodes (id, label, properties) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET label = excluded.label, properties = excluded.properties`,
		id, label, properties,
	)
	return err
}

func (k *KnowledgeDB) AddEdge(sourceID, targetID, relation string, weight float64) error {
	if weight <= 0 {
		weight = 1.0
	}
	_, err := k.db.Exec(
		`INSERT INTO edges (source_id, target_id, relation, weight) VALUES (?, ?, ?, ?) ON CONFLICT(source_id, target_id, relation) DO UPDATE SET weight = excluded.weight`,
		sourceID, targetID, relation, weight,
	)
	return err
}
