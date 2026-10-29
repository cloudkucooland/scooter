package knowledge

import (
	"database/sql"
	"encoding/binary"
	"errors"
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
	ID             string    `json:"id"`
	DocumentID     string    `json:"document_id"`
	ChunkIndex     int       `json:"chunk_index"`
	Content        string    `json:"content"`
	EmbeddingModel string    `json:"embedding_model,omitempty"`
	Embedding      []float32 `json:"embedding,omitempty"`
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
	db             *sql.DB
	EmbedFunc      func(string) ([]float32, error)
	QueryEmbedFunc func(string) ([]float32, error)
	EmbeddingModel string
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
		embedding_model TEXT,
		embedding BLOB,
		FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
	);

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

	_, _ = k.db.Exec(`ALTER TABLE chunks ADD COLUMN embedding_model TEXT;`)
	return nil
}

func serializeEmbedding(vec []float32) []byte {
	if len(vec) == 0 {
		return nil
	}
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func deserializeEmbedding(buf []byte) []float32 {
	if len(buf)%4 != 0 || len(buf) == 0 {
		return nil
	}
	vec := make([]float32, len(buf)/4)
	for i := range vec {
		bits := binary.LittleEndian.Uint32(buf[i*4:])
		vec[i] = math.Float32frombits(bits)
	}
	return vec
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	val := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	if math.IsNaN(val) {
		return 0
	}
	return val
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
	docID := fmt.Sprintf("doc_%d", time.Now().UnixNano())
	now := time.Now().Format(time.RFC3339)

	tx, err := k.db.Begin()
	if err != nil {
		return "", err
	}
	defer tx.Rollback()

	_, err = tx.Exec(
		`INSERT INTO documents (id, title, source_path, collection, content, metadata, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)`,
		docID, title, sourcePath, collection, content, "{}", now,
	)
	if err != nil {
		return "", err
	}

	chunks := splitIntoChunks(content, 1000)
	for i, chunkContent := range chunks {
		chunkID := fmt.Sprintf("%s_chunk_%d", docID, i)
		var embBlob []byte
		embModel := ""
		if k.EmbedFunc != nil {
			vec, err := k.EmbedFunc(chunkContent)
			if err != nil {
				return "", fmt.Errorf("embed chunk %d: %w", i, err)
			}
			if len(vec) == 0 {
				return "", fmt.Errorf("embed chunk %d: embedding is empty", i)
			}
			embBlob = serializeEmbedding(vec)
			embModel = k.EmbeddingModel
			if embModel == "" {
				embModel = "default-embedding"
			}
		}

		_, err = tx.Exec(
			`INSERT INTO chunks (id, document_id, chunk_index, content, embedding_model, embedding) VALUES (?, ?, ?, ?, ?, ?)`,
			chunkID, docID, i, chunkContent, embModel, embBlob,
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
	}

	return docID, tx.Commit()
}

func (k *KnowledgeDB) IngestFile(filePath, collection string) (string, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}
	title := filepath.Base(filePath)
	return k.IngestText(title, string(data), collection, filePath)
}

func (k *KnowledgeDB) Search(query string, collection string, limit int) ([]SearchResult, error) {
	if limit <= 0 {
		limit = 5
	}

	var rows *sql.Rows
	var err error
	if collection != "" {
		rows, err = k.db.Query(`
			SELECT c.id, d.id, d.title, d.collection, c.content, f.rank
			FROM chunks_fts f
			JOIN chunks c ON c.id = f.chunk_id
			JOIN documents d ON d.id = c.document_id
			WHERE chunks_fts MATCH ? AND d.collection = ?
			ORDER BY f.rank
			LIMIT ?
		`, query, collection, limit)
	} else {
		rows, err = k.db.Query(`
			SELECT c.id, d.id, d.title, d.collection, c.content, f.rank
			FROM chunks_fts f
			JOIN chunks c ON c.id = f.chunk_id
			JOIN documents d ON d.id = c.document_id
			WHERE chunks_fts MATCH ?
			ORDER BY f.rank
			LIMIT ?
		`, query, limit)
	}
	if err != nil {
		if !strings.Contains(strings.ToLower(err.Error()), "fts5:") {
			return nil, err
		}
		return k.searchLike(query, collection, limit)
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var sr SearchResult
		var rank float64
		if err := rows.Scan(&sr.ChunkID, &sr.DocumentID, &sr.Title, &sr.Collection, &sr.Content, &rank); err != nil {
			return nil, err
		}
		sr.Score = rank
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

func (k *KnowledgeDB) VectorSearch(queryVec []float32, embeddingModel string, collection string, limit int) ([]SearchResult, error) {
	if limit <= 0 {
		limit = 5
	}

	var rows *sql.Rows
	var err error
	if collection != "" {
		rows, err = k.db.Query(`
			SELECT c.id, c.document_id, c.embedding
			FROM chunks c
			JOIN documents d ON d.id = c.document_id
			WHERE c.embedding_model = ? AND c.embedding IS NOT NULL AND d.collection = ?
		`, embeddingModel, collection)
	} else {
		rows, err = k.db.Query(`
			SELECT c.id, c.document_id, c.embedding
			FROM chunks c
			WHERE c.embedding_model = ? AND c.embedding IS NOT NULL
		`, embeddingModel)
	}
	if err != nil {
		return nil, err
	}

	type candidate struct {
		chunkID    string
		documentID string
		score      float64
	}

	var candidates []candidate
	for rows.Next() {
		var chunkID, docID string
		var embBlob []byte
		if err := rows.Scan(&chunkID, &docID, &embBlob); err != nil {
			rows.Close()
			return nil, err
		}
		vec := deserializeEmbedding(embBlob)
		score := cosineSimilarity(queryVec, vec)
		candidates = append(candidates, candidate{
			chunkID:    chunkID,
			documentID: docID,
			score:      score,
		})
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return nil, err
	}
	if err := rows.Close(); err != nil {
		return nil, err
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	if len(candidates) > limit {
		candidates = candidates[:limit]
	}

	if len(candidates) == 0 {
		return nil, nil
	}

	chunkScores := make(map[string]float64, len(candidates))
	ids := make([]any, len(candidates))
	placeholders := make([]string, len(candidates))
	for i, c := range candidates {
		chunkScores[c.chunkID] = c.score
		ids[i] = c.chunkID
		placeholders[i] = "?"
	}

	query := fmt.Sprintf(`
		SELECT c.id, d.id, d.title, d.collection, c.content
		FROM chunks c
		JOIN documents d ON d.id = c.document_id
		WHERE c.id IN (%s)
	`, strings.Join(placeholders, ","))

	batchRows, err := k.db.Query(query, ids...)
	if err != nil {
		return nil, err
	}
	defer batchRows.Close()

	var results []SearchResult
	for batchRows.Next() {
		var sr SearchResult
		if err := batchRows.Scan(&sr.ChunkID, &sr.DocumentID, &sr.Title, &sr.Collection, &sr.Content); err != nil {
			return nil, err
		}
		sr.Score = chunkScores[sr.ChunkID]
		results = append(results, sr)
	}
	if err := batchRows.Err(); err != nil {
		return nil, err
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

func (k *KnowledgeDB) SearchHybrid(query string, collection string, limit int) ([]SearchResult, error) {
	if limit <= 0 {
		limit = 5
	}

	ftsRes, ftsErr := k.Search(query, collection, limit*2)

	var vecRes []SearchResult
	var embModel string
	modelErr := k.db.QueryRow(`SELECT embedding_model FROM chunks WHERE embedding_model != '' AND embedding IS NOT NULL LIMIT 1`).Scan(&embModel)
	var vectorErr error

	embedQuery := k.QueryEmbedFunc
	if embedQuery == nil {
		embedQuery = k.EmbedFunc
	}

	if modelErr != nil && !errors.Is(modelErr, sql.ErrNoRows) {
		vectorErr = fmt.Errorf("find embedding model: %w", modelErr)
	} else if embModel != "" {
		if embedQuery == nil {
			vectorErr = errors.New("embedding function is not configured")
		} else {
			qVec, err := embedQuery(query)
			if err != nil {
				vectorErr = fmt.Errorf("embed search query: %w", err)
			} else if len(qVec) == 0 {
				vectorErr = errors.New("query embedding is empty")
			} else {
				vecRes, vectorErr = k.VectorSearch(qVec, embModel, collection, limit*2)
			}
		}
	}

	if len(ftsRes) == 0 && len(vecRes) == 0 {
		results, likeErr := k.searchLike(query, collection, limit)
		if ftsErr != nil && (vectorErr != nil || errors.Is(modelErr, sql.ErrNoRows)) {
			return nil, errors.Join(fmt.Errorf("full-text search: %w", ftsErr), vectorErr, fmt.Errorf("LIKE search: %w", likeErr))
		}
		return results, likeErr
	}

	scores := make(map[string]float64)
	docMap := make(map[string]SearchResult)
	const kRRF = 60.0

	for rank, r := range ftsRes {
		id := r.ChunkID
		docMap[id] = r
		scores[id] += 1.0 / (kRRF + float64(rank+1))
	}

	for rank, r := range vecRes {
		id := r.ChunkID
		if _, exists := docMap[id]; !exists {
			docMap[id] = r
		}
		scores[id] += 1.0 / (kRRF + float64(rank+1))
	}

	type scoredDoc struct {
		id    string
		score float64
	}

	var combined []scoredDoc
	for id, score := range scores {
		combined = append(combined, scoredDoc{id: id, score: score})
	}

	sort.Slice(combined, func(i, j int) bool {
		return combined[i].score > combined[j].score
	})

	var results []SearchResult
	for i := 0; i < len(combined) && i < limit; i++ {
		res := docMap[combined[i].id]
		res.Score = combined[i].score
		results = append(results, res)
	}

	return results, nil
}

func (k *KnowledgeDB) ReembedChunks(embeddingModel string, embedFunc func(string) ([]float32, error)) error {
	if embedFunc == nil {
		return errors.New("embedding function is required")
	}
	rows, err := k.db.Query(`SELECT id, content FROM chunks`)
	if err != nil {
		return err
	}

	type pendingChunk struct {
		id      string
		content string
	}
	var pending []pendingChunk
	for rows.Next() {
		var p pendingChunk
		if err := rows.Scan(&p.id, &p.content); err == nil {
			pending = append(pending, p)
		} else {
			rows.Close()
			return err
		}
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return err
	}
	if err := rows.Close(); err != nil {
		return err
	}

	type chunkUpdate struct {
		id  string
		vec []float32
	}
	var updates []chunkUpdate
	for _, item := range pending {
		vec, err := embedFunc(item.content)
		if err != nil {
			return fmt.Errorf("embed chunk %s: %w", item.id, err)
		}
		if len(vec) == 0 {
			return fmt.Errorf("embed chunk %s: embedding is empty", item.id)
		}
		updates = append(updates, chunkUpdate{id: item.id, vec: vec})
	}

	tx, err := k.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(`UPDATE chunks SET embedding_model = ?, embedding = ? WHERE id = ?`)
	if err != nil {
		return err
	}
	defer stmt.Close()

	for _, u := range updates {
		blob := serializeEmbedding(u.vec)
		if _, err := stmt.Exec(embeddingModel, blob, u.id); err != nil {
			return err
		}
	}

	return tx.Commit()
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
		rows, err = k.db.Query(`SELECT source_id, target_id, relation, weight FROM edges WHERE source_id = ? AND relation = ?`, sourceID, relation)
	} else {
		rows, err = k.db.Query(`SELECT source_id, target_id, relation, weight FROM edges WHERE source_id = ?`, sourceID)
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
	if properties == "" {
		properties = "{}"
	}
	_, err := k.db.Exec(`INSERT OR REPLACE INTO nodes (id, label, properties) VALUES (?, ?, ?)`, id, label, properties)
	return err
}

func (k *KnowledgeDB) AddEdge(sourceID, targetID, relation string, weight float64) error {
	if weight == 0 {
		weight = 1.0
	}
	_, err := k.db.Exec(`INSERT OR REPLACE INTO edges (source_id, target_id, relation, weight) VALUES (?, ?, ?, ?)`, sourceID, targetID, relation, weight)
	return err
}

func splitIntoChunks(text string, maxChars int) []string {
	paragraphs := strings.Split(text, "\n\n")
	var chunks []string
	var current strings.Builder
	var currentRunes int

	for _, p := range paragraphs {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		pRunes := []rune(p)
		if len(pRunes) > maxChars {
			if current.Len() > 0 {
				chunks = append(chunks, strings.TrimSpace(current.String()))
				current.Reset()
				currentRunes = 0
			}
			for i := 0; i < len(pRunes); i += maxChars {
				end := i + maxChars
				if end > len(pRunes) {
					end = len(pRunes)
				}
				chunks = append(chunks, strings.TrimSpace(string(pRunes[i:end])))
			}
			continue
		}

		extra := 0
		if current.Len() > 0 {
			extra = 2
		}
		if currentRunes+len(pRunes)+extra > maxChars && current.Len() > 0 {
			chunks = append(chunks, strings.TrimSpace(current.String()))
			current.Reset()
			currentRunes = 0
			extra = 0
		}
		if current.Len() > 0 {
			current.WriteString("\n\n")
			currentRunes += 2
		}
		current.WriteString(p)
		currentRunes += len(pRunes)
	}
	if current.Len() > 0 {
		chunks = append(chunks, strings.TrimSpace(current.String()))
	}
	return chunks
}
