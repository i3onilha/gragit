package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/linka-ai/gragit/internal/rag/chunking"
	"github.com/linka-ai/gragit/internal/rag/config"
	"github.com/linka-ai/gragit/internal/rag/embeddings"
	"github.com/linka-ai/gragit/internal/rag/gitrepo"
	"github.com/linka-ai/gragit/internal/rag/ingestion"
	"github.com/linka-ai/gragit/internal/rag/vectorstore"
	"github.com/spf13/cobra"
)

var (
	remoteName string
	branchName string
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("")

	if err := newRootCmd().Execute(); err != nil {
		os.Exit(1)
	}
}

func newRootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   "gragit",
		Short: "Index Git repositories into a local vector store",
	}

	root.AddCommand(newIngestCmd())
	return root
}

func newIngestCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "ingest",
		Short: "Clone a remote branch and build a FAISS index",
		Long: `Detect a git remote from the current directory, clone or refresh the chosen
branch under ~/.gragit/repos/<host>/<owner>/<repo>/<branch>, index the clone,
and save the FAISS bundle under ~/.gragit/indexes/<host>/<owner>/<repo>/<branch>/<model>.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if err := runIngest(remoteName, branchName); err != nil {
				log.Printf("ERROR ingest: %v", err)
				return err
			}
			return nil
		},
	}

	cmd.Flags().StringVarP(&remoteName, "remote", "r", "origin",
		"Git remote to use (e.g. origin, upstream)")
	cmd.Flags().StringVarP(&branchName, "branch", "b", "develop",
		"Branch to clone and index (synced to remotes/<remote>/<branch>)")

	return cmd
}

func runIngest(remote, branch string) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	info, err := gitrepo.ResolveFromCWD(remote, branch)
	if err != nil {
		return err
	}

	indexPath, err := gitrepo.IndexDir(info, cfg.EmbeddingModel)
	if err != nil {
		return err
	}

	log.Printf("INFO git: %s/%s/%s branch %s", info.Host, info.User, info.Repository, info.Branch)
	log.Printf("INFO git: remote %s -> %s", info.RemoteName, info.RemoteURL)
	log.Printf("INFO git: syncing remotes/%s/%s", info.RemoteName, info.Branch)

	clonePath, err := gitrepo.Sync(info)
	if err != nil {
		return err
	}
	log.Printf("INFO git: repository ready at %s", clonePath)

	commitSHA, err := gitrepo.CommitSHA(clonePath)
	if err != nil {
		return fmt.Errorf("read commit sha: %w", err)
	}

	log.Printf("INFO pipeline: starting ingestion for %s @ %s", clonePath, commitSHA[:min(7, len(commitSHA))])
	docs, err := ingestion.LoadDocuments([]string{clonePath})
	if err != nil {
		return err
	}
	if len(docs) == 0 {
		return fmt.Errorf("no documents loaded from cloned repository")
	}

	chunks := chunking.SplitDocuments(docs, cfg.ChunkSize, cfg.ChunkOverlap)
	if len(chunks) == 0 {
		return fmt.Errorf("no chunks produced from documents")
	}

	texts := make([]string, len(chunks))
	for i, chunk := range chunks {
		texts[i] = chunk.PageContent
	}

	ctx := context.Background()
	embedder, err := embeddings.New(ctx, cfg)
	if err != nil {
		return err
	}
	defer embedder.Close()

	log.Printf("INFO pipeline: embedding %d chunk(s)", len(texts))
	vectors, err := embedder.EmbedTexts(ctx, texts)
	if err != nil {
		return err
	}

	if err := vectorstore.Save(indexPath, chunks, vectors); err != nil {
		return err
	}

	dimension := 0
	if len(vectors) > 0 {
		dimension = len(vectors[0])
	}
	if err := gitrepo.WriteIndexManifest(indexPath, gitrepo.IndexManifest{
		RemoteURL:          info.RemoteURL,
		Host:               info.Host,
		Owner:              info.User,
		Repository:         info.Repository,
		Branch:             info.Branch,
		CommitSHA:          commitSHA,
		EmbeddingModel:     cfg.EmbeddingModel,
		ChunkSize:          cfg.ChunkSize,
		ChunkOverlap:       cfg.ChunkOverlap,
		SourceClonePath:    clonePath,
		VectorCount:        len(vectors),
		EmbeddingDimension: dimension,
	}); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}

	fmt.Printf("Indexed repository. Clone: %s\n", clonePath)
	fmt.Printf("FAISS saved to: %s\n", indexPath)
	return nil
}
