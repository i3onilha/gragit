package embeddings

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync/atomic"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
	"golang.org/x/sync/errgroup"

	"github.com/i3onilha/ragcode/internal/rag/config"
	"github.com/i3onilha/ragcode/internal/rag/gitrepo"
)

const (
	// all-MiniLM-L6-v2 max sequence is 512 tokens; keep a safe character budget.
	maxEmbedRunes = 600
)

// Embedder generates normalized sentence embeddings locally via hugot.
type Embedder struct {
	session   *hugot.Session
	pipeline  *pipelines.FeatureExtractionPipeline
	batchSize int
	workers   int
}

// New creates an embedder for the configured model.
func New(ctx context.Context, cfg config.Config) (*Embedder, error) {
	session, err := hugot.NewGoSession(ctx)
	if err != nil {
		return nil, fmt.Errorf("create hugot session: %w", err)
	}

	modelsDir, err := gitrepo.ModelsCacheDir()
	if err != nil {
		session.Destroy()
		return nil, err
	}
	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		session.Destroy()
		return nil, err
	}

	modelName := cfg.HuggingFaceONNXModel()
	log.Printf("INFO embeddings: downloading/loading model %s", modelName)
	modelPath, err := hugot.DownloadModel(ctx, modelName, modelsDir, hugot.NewDownloadOptions())
	if err != nil {
		session.Destroy()
		return nil, fmt.Errorf("download model %s: %w", modelName, err)
	}

	pipelineConfig := hugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "ingestEmbeddings",
		OnnxFilename: "model.onnx",
		Options: []hugot.FeatureExtractionOption{
			pipelines.WithNormalization(),
		},
	}
	pipeline, err := hugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		session.Destroy()
		return nil, fmt.Errorf("create embedding pipeline: %w", err)
	}

	batchSize := cfg.EmbedBatchSize
	if batchSize < 1 {
		batchSize = 1
	}
	workers := cfg.EmbedWorkers
	if workers < 1 {
		workers = 1
	}

	log.Printf("INFO embeddings: batch size %d, workers %d", batchSize, workers)

	return &Embedder{
		session:   session,
		pipeline:  pipeline,
		batchSize: batchSize,
		workers:   workers,
	}, nil
}

// Close releases hugot resources.
func (e *Embedder) Close() error {
	if e.session == nil {
		return nil
	}
	return e.session.Destroy()
}

// EmbedQuery returns a normalized embedding vector for a single query string.
func (e *Embedder) EmbedQuery(ctx context.Context, query string) ([]float32, error) {
	vectors, err := e.EmbedTexts(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, fmt.Errorf("empty embedding for query")
	}
	return vectors[0], nil
}

// EmbedTexts returns one normalized embedding vector per input string.
func (e *Embedder) EmbedTexts(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	prepared := make([]string, len(texts))
	for i, text := range texts {
		prepared[i] = truncateForEmbedding(text)
	}

	vectors := make([][]float32, len(prepared))
	if e.workers == 1 {
		return e.embedSequential(ctx, prepared, vectors)
	}
	return e.embedConcurrent(ctx, prepared, vectors)
}

func (e *Embedder) embedSequential(ctx context.Context, prepared []string, vectors [][]float32) ([][]float32, error) {
	for start := 0; start < len(prepared); start += e.batchSize {
		end := start + e.batchSize
		if end > len(prepared) {
			end = len(prepared)
		}
		batchVectors, err := e.embedBatch(ctx, prepared[start:end])
		if err != nil {
			return nil, fmt.Errorf("embed batch %d-%d: %w", start, end, err)
		}
		copy(vectors[start:end], batchVectors)
		log.Printf("INFO embeddings: %d/%d chunk(s) embedded", end, len(prepared))
	}
	return vectors, nil
}

func (e *Embedder) embedConcurrent(ctx context.Context, prepared []string, vectors [][]float32) ([][]float32, error) {
	var completed atomic.Int64
	total := int64(len(prepared))

	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(e.workers)

	for start := 0; start < len(prepared); start += e.batchSize {
		start := start
		end := start + e.batchSize
		if end > len(prepared) {
			end = len(prepared)
		}

		g.Go(func() error {
			batchVectors, err := e.embedBatch(ctx, prepared[start:end])
			if err != nil {
				return fmt.Errorf("embed batch %d-%d: %w", start, end, err)
			}
			copy(vectors[start:end], batchVectors)

			done := completed.Add(int64(end - start))
			log.Printf("INFO embeddings: %d/%d chunk(s) embedded", done, total)
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}
	return vectors, nil
}

func truncateForEmbedding(text string) string {
	runes := []rune(text)
	if len(runes) <= maxEmbedRunes {
		return text
	}
	return string(runes[:maxEmbedRunes])
}

func (e *Embedder) embedBatch(ctx context.Context, batch []string) ([][]float32, error) {
	out, err := e.pipeline.RunPipeline(ctx, batch)
	if err == nil {
		return out.Embeddings, nil
	}

	// hugot's Go backend can fail when mixed-length texts share a batch; fall back per item.
	if len(batch) == 1 {
		return nil, err
	}

	vectors := make([][]float32, 0, len(batch))
	for _, text := range batch {
		single, singleErr := e.pipeline.RunPipeline(ctx, []string{text})
		if singleErr != nil {
			return nil, singleErr
		}
		vectors = append(vectors, single.Embeddings...)
	}
	return vectors, nil
}
