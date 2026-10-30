package knowledge

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
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
