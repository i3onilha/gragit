package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

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
and save the FAISS bundle under ~/.gragit/indexes/<host>/<owner>/<repo>/<branch>.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			settings, err := resolveIngestSettings(remoteName, branchName)
			if err != nil {
				log.Printf("ERROR ingest: %v", err)
				return err
			}

			if !ingestFlagsChanged(cmd) {
				printIngestSettings(settings)
				ok, err := confirmIngest(os.Stdin, os.Stdout)
				if err != nil {
					log.Printf("ERROR ingest: %v", err)
					return err
				}
				if !ok {
					fmt.Println("Cancelled.")
					return nil
				}
			}

			if err := runIngest(settings); err != nil {
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

type ingestSettings struct {
	info      gitrepo.Info
	cfg       config.Config
	clonePath string
	indexPath string
}

func ingestFlagsChanged(cmd *cobra.Command) bool {
	return cmd.Flags().Changed("remote") || cmd.Flags().Changed("branch")
}

func resolveIngestSettings(remote, branch string) (ingestSettings, error) {
	cfg, err := config.Load()
	if err != nil {
		return ingestSettings{}, err
	}

	info, err := gitrepo.ResolveFromCWD(remote, branch)
	if err != nil {
		return ingestSettings{}, err
	}

	clonePath, err := gitrepo.RepositoryDir(info)
	if err != nil {
		return ingestSettings{}, err
	}

	indexPath, err := gitrepo.IndexDir(info)
	if err != nil {
		return ingestSettings{}, err
	}

	return ingestSettings{
		info:      info,
		cfg:       cfg,
		clonePath: clonePath,
		indexPath: indexPath,
	}, nil
}

func printIngestSettings(s ingestSettings) {
	fmt.Println("Ingest will run with these parameters:")
	fmt.Printf("  remote (--remote, -r):     %s\n", s.info.RemoteName)
	fmt.Printf("  branch (--branch, -b):     %s\n", s.info.Branch)
	fmt.Printf("  remote URL:                %s\n", s.info.RemoteURL)
	fmt.Printf("  repository:                %s/%s/%s\n", s.info.Host, s.info.User, s.info.Repository)
	fmt.Printf("  clone path:                %s\n", s.clonePath)
	fmt.Printf("  index path:                %s\n", s.indexPath)
	fmt.Printf("  EMBEDDING_MODEL:           %s\n", s.cfg.EmbeddingModel)
	fmt.Printf("  CHUNK_SIZE:                %d\n", s.cfg.ChunkSize)
	fmt.Printf("  CHUNK_OVERLAP:             %d\n", s.cfg.ChunkOverlap)
	fmt.Printf("  EMBED_BATCH_SIZE:          %d\n", s.cfg.EmbedBatchSize)
	fmt.Printf("  EMBED_WORKERS:             %d\n", s.cfg.EmbedWorkers)
}

func confirmIngest(in io.Reader, out io.Writer) (bool, error) {
	if f, ok := in.(*os.File); ok && !isTerminal(f) {
		return false, fmt.Errorf("non-interactive stdin; pass --remote (-r) and/or --branch (-b) to skip confirmation")
	}

	fmt.Fprint(out, "Proceed with ingest? [y/N]: ")
	answer, err := bufio.NewReader(in).ReadString('\n')
	if err != nil && err != io.EOF {
		return false, fmt.Errorf("read confirmation: %w", err)
	}
	switch strings.ToLower(strings.TrimSpace(answer)) {
	case "y", "yes":
		return true, nil
	default:
		return false, nil
	}
}

func isTerminal(f *os.File) bool {
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

func runIngest(s ingestSettings) error {
	info := s.info
	cfg := s.cfg
	indexPath := s.indexPath

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

	if gitrepo.IndexBundleComplete(indexPath) {
		manifest, err := gitrepo.ReadIndexManifest(indexPath)
		if err == nil && gitrepo.IndexMatchesSettings(
			manifest,
			commitSHA,
			cfg.EmbeddingModel,
			cfg.ChunkSize,
			cfg.ChunkOverlap,
		) {
			log.Printf("INFO pipeline: index up to date (%d vectors, model %s)", manifest.VectorCount, manifest.EmbeddingModel)
			fmt.Printf("Index already current. Clone: %s\n", clonePath)
			fmt.Printf("FAISS index: %s\n", indexPath)
			return nil
		}
	}

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

