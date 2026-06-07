package pipeline

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/i3onilha/gragit/internal/rag/config"
	"github.com/i3onilha/gragit/internal/rag/embeddings"
	"github.com/i3onilha/gragit/internal/rag/gitrepo"
	"github.com/i3onilha/gragit/internal/rag/generator"
	"github.com/i3onilha/gragit/internal/rag/retriever"
	"github.com/i3onilha/gragit/internal/rag/vectorstore"
)

// AskResult is the payload returned by Pipeline.Ask.
type AskResult struct {
	Answer          string
	Sources         []string
	RetrievedChunks int
	RateLimited     bool
}

// Pipeline orchestrates retrieval and grounded answer generation.
type Pipeline struct {
	cfg      config.Config
	embedder *embeddings.Embedder
	index    *vectorstore.Index
}

// New builds a pipeline from environment configuration and the given index path.
func New(ctx context.Context, indexPath string) (*Pipeline, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, err
	}

	log.Printf("INFO pipeline: RAG LLM model: %s", cfg.LLMModel)

	if manifest, err := gitrepo.ReadIndexManifest(indexPath); err == nil && manifest.EmbeddingModel != "" {
		cfg.EmbeddingModel = manifest.EmbeddingModel
	}

	embedder, err := embeddings.New(ctx, cfg)
	if err != nil {
		return nil, err
	}

	index, err := vectorstore.Load(indexPath)
	if err != nil {
		embedder.Close()
		return nil, err
	}

	return &Pipeline{
		cfg:      cfg,
		embedder: embedder,
		index:    index,
	}, nil
}

// Close releases embedding resources.
func (p *Pipeline) Close() error {
	if p.embedder == nil {
		return nil
	}
	return p.embedder.Close()
}

// Ask embeds the question, retrieves top-k chunks, and generates a grounded answer.
func (p *Pipeline) Ask(ctx context.Context, question string) (AskResult, error) {
	question = strings.TrimSpace(question)
	if question == "" {
		return AskResult{}, fmt.Errorf("question must be non-empty")
	}

	log.Printf("INFO pipeline: question: %s", truncate(question, 200))
	chunks, err := retriever.Retrieve(ctx, question, p.index, p.embedder, p.cfg.TopK)
	if err != nil {
		return AskResult{}, err
	}

	gen, err := generator.GenerateAnswer(ctx, question, chunks, p.cfg)
	if err != nil {
		return AskResult{}, err
	}

	return AskResult{
		Answer:          gen.Answer,
		Sources:         gen.Sources,
		RetrievedChunks: len(chunks),
		RateLimited:     gen.RateLimited,
	}, nil
}

func truncate(s string, max int) string {
	runes := []rune(s)
	if len(runes) <= max {
		return s
	}
	return string(runes[:max])
}
