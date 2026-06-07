package config

import (
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/joho/godotenv"
)

func defaultEmbedWorkers() int {
	n := runtime.NumCPU()
	if n > 4 {
		return 4
	}
	if n < 1 {
		return 1
	}
	return n
}

const defaultOpenRouterBaseURL = "https://openrouter.ai/api/v1"

// Config holds tunable RAG settings for ingestion and ask.
type Config struct {
	ChunkSize        int
	ChunkOverlap     int
	TopK             int
	EmbedBatchSize   int
	EmbedWorkers     int
	EmbeddingModel   string
	OpenRouterAPIKey string
	LLMModel         string
	LLMBaseURL       string
}

// Load reads environment variables with the same defaults as the Python RAG app.
func Load() (Config, error) {
	_ = godotenv.Load(".env")
	_ = godotenv.Overload(filepath.Join("src", ".env"))

	apiKey := strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY"))
	if apiKey == "" {
		if rootVals, err := godotenv.Read(".env"); err == nil {
			if fallback := strings.TrimSpace(rootVals["OPENROUTER_API_KEY"]); fallback != "" {
				apiKey = fallback
				_ = os.Setenv("OPENROUTER_API_KEY", fallback)
			}
		}
	}

	cfg := Config{
		ChunkSize:        envInt("CHUNK_SIZE", 1000),
		ChunkOverlap:     envInt("CHUNK_OVERLAP", 200),
		TopK:             envInt("TOP_K", 5),
		EmbedBatchSize:   envInt("EMBED_BATCH_SIZE", 4),
		EmbedWorkers:     envInt("EMBED_WORKERS", 0),
		EmbeddingModel:   envString("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
		OpenRouterAPIKey: apiKey,
		LLMBaseURL:       strings.TrimRight(envString("OPENROUTER_BASE_URL", defaultOpenRouterBaseURL), "/"),
	}
	if ragModel := strings.TrimSpace(os.Getenv("OPENROUTER_RAG_MODEL")); ragModel != "" {
		cfg.LLMModel = ragModel
	} else {
		cfg.LLMModel = envString("OPENROUTER_MODEL", "google/gemma-3-12b-it:free")
	}
	if cfg.EmbedBatchSize < 1 {
		cfg.EmbedBatchSize = 1
	}
	if cfg.EmbedWorkers < 1 {
		cfg.EmbedWorkers = defaultEmbedWorkers()
	}

	return cfg, nil
}

// HuggingFaceONNXModel maps sentence-transformers names to ONNX models usable by hugot.
func (c Config) HuggingFaceONNXModel() string {
	model := strings.TrimSpace(c.EmbeddingModel)
	switch model {
	case "all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2":
		return "KnightsAnalytics/all-MiniLM-L6-v2"
	default:
		return model
	}
}

func envString(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return n
}
