package retriever

import (
	"context"
	"log"

	"github.com/i3onilha/ragcode/internal/rag/document"
	"github.com/i3onilha/ragcode/internal/rag/embeddings"
	"github.com/i3onilha/ragcode/internal/rag/vectorstore"
)

// Retrieve embeds the query and returns the top-k most similar chunks.
func Retrieve(
	ctx context.Context,
	query string,
	index *vectorstore.Index,
	embedder *embeddings.Embedder,
	topK int,
) ([]document.Document, error) {
	vector, err := embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, err
	}

	k := topK
	if k < 1 {
		k = 1
	}
	docs := index.Search(vector, k)
	log.Printf("INFO retriever: retrieved %d chunk(s)", len(docs))
	return docs, nil
}
