package embeddings

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"

	"github.com/linka-ai/gragit/internal/rag/config"
	"github.com/linka-ai/gragit/internal/rag/gitrepo"
)

const (
	embedBatchSize = 16
	// all-MiniLM-L6-v2 max sequence is 512 tokens; keep a safe character budget.
	maxEmbedRunes = 600
)

// Embedder generates normalized sentence embeddings locally via hugot.
type Embedder struct {
	session  *hugot.Session
	pipeline *pipelines.FeatureExtractionPipeline
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

	return &Embedder{
		session:  session,
		pipeline: pipeline,
	}, nil
}

// Close releases hugot resources.
func (e *Embedder) Close() error {
	if e.session == nil {
		return nil
	}
	return e.session.Destroy()
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

	vectors := make([][]float32, 0, len(texts))
	for start := 0; start < len(prepared); start += embedBatchSize {
		end := start + embedBatchSize
		if end > len(prepared) {
			end = len(prepared)
		}
		batch := prepared[start:end]
		batchVectors, err := e.embedBatch(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("embed batch %d-%d: %w", start, end, err)
		}
		vectors = append(vectors, batchVectors...)
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
