package vectorstore

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"github.com/google/uuid"

	"github.com/i3onilha/gragit/internal/rag/document"
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

// Index is an in-memory flat L2 index loaded from docstore.json + vectors.bin.
type Index struct {
	dimension int
	vectors   [][]float32
	documents []document.Document
}

// Load reads a Go-produced FAISS-compatible index bundle from disk.
func Load(indexPath string) (*Index, error) {
	docstorePath := filepath.Join(indexPath, docstoreFile)
	vectorsPath := filepath.Join(indexPath, vectorsFile)

	docstoreBytes, err := os.ReadFile(docstorePath)
	if err != nil {
		return nil, fmt.Errorf("read docstore: %w", err)
	}
	vectorBytes, err := os.ReadFile(vectorsPath)
	if err != nil {
		return nil, fmt.Errorf("read vectors: %w", err)
	}

	var payload docstorePayload
	if err := json.Unmarshal(docstoreBytes, &payload); err != nil {
		return nil, fmt.Errorf("parse docstore: %w", err)
	}

	expectedBytes := payload.Count * payload.Dimension * 4
	if len(vectorBytes) != expectedBytes {
		return nil, fmt.Errorf(
			"vectors.bin size mismatch: got %d bytes, expected %d",
			len(vectorBytes), expectedBytes,
		)
	}

	flat := make([]float32, payload.Count*payload.Dimension)
	for i := range flat {
		flat[i] = math.Float32frombits(binary.LittleEndian.Uint32(vectorBytes[i*4 : (i+1)*4]))
	}

	vectors := make([][]float32, payload.Count)
	documents := make([]document.Document, payload.Count)
	for i := 0; i < payload.Count; i++ {
		start := i * payload.Dimension
		end := start + payload.Dimension
		vectors[i] = flat[start:end]

		docID, ok := payload.IndexToDocstoreID[strconv.Itoa(i)]
		if !ok {
			return nil, fmt.Errorf("missing docstore id for index position %d", i)
		}
		stored, ok := payload.Docs[docID]
		if !ok {
			return nil, fmt.Errorf("missing document for id %s", docID)
		}
		documents[i] = document.Document{
			PageContent: stored.PageContent,
			Metadata:    stored.Metadata,
		}
	}

	log.Printf("INFO vectorstore: loaded %d vectors (dim=%d) from %s", payload.Count, payload.Dimension, indexPath)
	return &Index{
		dimension: payload.Dimension,
		vectors:   vectors,
		documents: documents,
	}, nil
}

type scoredDoc struct {
	score float64
	doc   document.Document
}

// Search returns up to topK documents ordered by ascending L2 distance.
func (idx *Index) Search(query []float32, topK int) []document.Document {
	if len(idx.vectors) == 0 {
		return nil
	}
	if topK < 1 {
		topK = 1
	}

	scored := make([]scoredDoc, len(idx.vectors))
	for i, vec := range idx.vectors {
		scored[i] = scoredDoc{
			score: l2Distance(query, vec),
			doc:   idx.documents[i],
		}
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score < scored[j].score
	})

	if topK > len(scored) {
		topK = len(scored)
	}
	out := make([]document.Document, topK)
	for i := 0; i < topK; i++ {
		out[i] = scored[i].doc
	}
	return out
}

func l2Distance(a, b []float32) float64 {
	var sum float64
	for i := range a {
		diff := float64(a[i]) - float64(b[i])
		sum += diff * diff
	}
	return sum
}

// Save writes a Go-produced FAISS-compatible index bundle.
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
