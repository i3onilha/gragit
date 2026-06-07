package generator

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"github.com/i3onilha/gragit/internal/rag/config"
	"github.com/i3onilha/gragit/internal/rag/document"
)

const ragPromptTemplate = `You are a helpful assistant that answers questions strictly based on the provided context.

Context:
%s

Question: %s

Instructions:
- Answer using ONLY the information in the context above.
- If the answer is not in the context, respond with: "I could not find an answer in the provided documents."
- At the end of your answer, list the sources you used under a "Sources:" heading. Use the source labels shown in the context (e.g. file paths).

Answer:`

const rateLimitMessage = `OpenRouter returned HTTP 429 (rate limited). Free models such as ` +
	`google/gemma-3-12b-it:free are often throttled upstream.

Fix: set OPENROUTER_RAG_MODEL or OPENROUTER_MODEL to another model in .env ` +
	`(for example openai/gpt-4o-mini), wait and retry, or add provider keys in ` +
	`OpenRouter integrations.`

// Result is the grounded answer payload returned by GenerateAnswer.
type Result struct {
	Answer      string
	Sources     []string
	ChunksUsed  int
	RateLimited bool
}

type chatRequest struct {
	Model       string        `json:"model"`
	Temperature float64       `json:"temperature"`
	Messages    []chatMessage `json:"messages"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// GenerateAnswer runs the chat model on retrieved chunks.
func GenerateAnswer(
	ctx context.Context,
	question string,
	chunks []document.Document,
	cfg config.Config,
) (Result, error) {
	if strings.TrimSpace(cfg.OpenRouterAPIKey) == "" {
		return Result{}, fmt.Errorf("OPENROUTER_API_KEY is not set. Add it to .env")
	}

	contextText := formatContext(chunks)
	sources := uniqueSources(chunks)
	prompt := fmt.Sprintf(ragPromptTemplate, contextText, question)

	log.Printf("INFO generator: calling LLM (%s) for grounded answer", cfg.LLMModel)
	body, err := json.Marshal(chatRequest{
		Model:       cfg.LLMModel,
		Temperature: 0.1,
		Messages: []chatMessage{
			{Role: "user", Content: prompt},
		},
	})
	if err != nil {
		return Result{}, err
	}

	url := cfg.LLMBaseURL + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return Result{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.OpenRouterAPIKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return Result{}, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return Result{}, err
	}

	if resp.StatusCode == http.StatusTooManyRequests {
		log.Printf("WARN generator: OpenRouter rate limit for model %s", cfg.LLMModel)
		return Result{
			Answer:      rateLimitMessage,
			Sources:     sources,
			ChunksUsed:  len(chunks),
			RateLimited: true,
		}, nil
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return Result{}, fmt.Errorf("openrouter chat completions: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}

	var parsed chatResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return Result{}, fmt.Errorf("decode openrouter response: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return Result{}, fmt.Errorf("openrouter returned no choices")
	}

	return Result{
		Answer:     strings.TrimSpace(parsed.Choices[0].Message.Content),
		Sources:    sources,
		ChunksUsed: len(chunks),
	}, nil
}

func formatContext(chunks []document.Document) string {
	if len(chunks) == 0 {
		return "(no context retrieved)"
	}

	var b strings.Builder
	for i, doc := range chunks {
		src := "unknown"
		if v, ok := doc.Metadata["source"].(string); ok && strings.TrimSpace(v) != "" {
			src = v
		}
		if i > 0 {
			b.WriteString("\n\n")
		}
		fmt.Fprintf(&b, "[%d] (source: %s)\n%s", i+1, src, strings.TrimSpace(doc.PageContent))
	}
	return b.String()
}

func uniqueSources(chunks []document.Document) []string {
	seen := make(map[string]struct{})
	out := make([]string, 0, len(chunks))
	for _, doc := range chunks {
		src, _ := doc.Metadata["source"].(string)
		src = strings.TrimSpace(src)
		if src == "" {
			continue
		}
		if _, ok := seen[src]; ok {
			continue
		}
		seen[src] = struct{}{}
		out = append(out, src)
	}
	return out
}
