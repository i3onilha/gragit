package vectorstore

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/google/uuid"

	"github.com/linka-ai/git-rag/internal/rag/document"
)

const (
	vectorsFile  = "vectors.bin"
	docstoreFile = "docstore.json"
)

type storedDoc struct {
	PageContent string         `json:"page_content"`
	Metadata    map[string]any `json:"metadata"`
}

type docstorePayload struct {
	Version             int                  `json:"version"`
	Metric              string               `json:"metric"`
	Dimension           int                  `json:"dimension"`
	Count               int                  `json:"count"`
	IndexToDocstoreID   map[string]string    `json:"index_to_docstore_id"`
	Docs                map[string]storedDoc `json:"docs"`
}

// Save writes a Go-produced FAISS-compatible index bundle for the Python ask command.
func Save(indexPath string, chunks []document.Document, vectors [][]float32) error {
	if len(chunks) == 0 {
		return fmt.Errorf("no chunks to index")
	}
	if len(chunks) != len(vectors) {
		return fmt.Errorf("chunk/vector count mismatch: %d chunks, %d vectors", len(chunks), len(vectors))
	}

	dim := len(vectors[0])
	for i, vec := range vectors {
		if len(vec) != dim {
			return fmt.Errorf("vector %d has dimension %d, expected %d", i, len(vec), dim)
		}
	}

	if err := os.MkdirAll(indexPath, 0o755); err != nil {
		return err
	}

	payload := docstorePayload{
		Version:           1,
		Metric:            "l2",
		Dimension:         dim,
		Count:             len(chunks),
		IndexToDocstoreID: make(map[string]string, len(chunks)),
		Docs:              make(map[string]storedDoc, len(chunks)),
	}

	flat := make([]float32, 0, len(chunks)*dim)
	for i, chunk := range chunks {
		id := uuid.NewString()
		payload.IndexToDocstoreID[fmt.Sprintf("%d", i)] = id
		payload.Docs[id] = storedDoc{
			PageContent: chunk.PageContent,
			Metadata:    chunk.Metadata,
		}
		flat = append(flat, vectors[i]...)
	}

	if err := writeVectors(filepath.Join(indexPath, vectorsFile), flat); err != nil {
		return err
	}
	if err := writeJSON(filepath.Join(indexPath, docstoreFile), payload); err != nil {
		return err
	}

	log.Printf("INFO vectorstore: saved %d vectors (dim=%d) to %s", len(chunks), dim, indexPath)
	return nil
}

func writeVectors(path string, values []float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	for _, v := range values {
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	return nil
}

func writeJSON(path string, payload any) error {
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
