package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/i3onilha/ragcode/internal/rag/config"
	"github.com/i3onilha/ragcode/internal/rag/gitrepo"
	"github.com/i3onilha/ragcode/internal/rag/pipeline"
	"github.com/spf13/cobra"
)

func newAskCmd() *cobra.Command {
	var question string

	cmd := &cobra.Command{
		Use:   "ask [question]",
		Short: "Ask a question about the indexed repository using an LLM",
		Long: `Load the FAISS index for the current git repository, retrieve relevant
chunks by semantic similarity, and generate a grounded answer via OpenRouter.

Omit the question for interactive mode. Requires OPENROUTER_API_KEY in .env.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) > 0 {
				question = strings.TrimSpace(strings.Join(args, " "))
			}

			settings, err := resolveAskSettings(remoteName, branchName)
			if err != nil {
				log.Printf("ERROR ask: %v", err)
				return err
			}

			if !gitrepo.IndexBundleComplete(settings.indexPath) {
				return fmt.Errorf("no index found at %s — run `ragcode ingest` first", settings.indexPath)
			}

			printAskSettings(settings)

			return runAsk(settings, question)
		},
	}

	cmd.Flags().StringVarP(&remoteName, "remote", "r", "origin",
		"Git remote to use (e.g. origin, upstream)")
	cmd.Flags().StringVarP(&branchName, "branch", "b", "develop",
		"Branch whose index to query")

	return cmd
}

type askSettings struct {
	info      gitrepo.Info
	cfg       config.Config
	indexPath string
}

func resolveAskSettings(remote, branch string) (askSettings, error) {
	cfg, err := config.Load()
	if err != nil {
		return askSettings{}, err
	}

	info, err := gitrepo.ResolveFromCWD(remote, branch)
	if err != nil {
		return askSettings{}, err
	}

	indexPath, err := gitrepo.IndexDir(info)
	if err != nil {
		return askSettings{}, err
	}

	return askSettings{
		info:      info,
		cfg:       cfg,
		indexPath: indexPath,
	}, nil
}

func printAskSettings(s askSettings) {
	fmt.Println("Ask will run with these parameters:")
	fmt.Printf("  remote (--remote, -r):     %s\n", s.info.RemoteName)
	fmt.Printf("  branch (--branch, -b):     %s\n", s.info.Branch)
	fmt.Printf("  repository:                %s/%s/%s\n", s.info.Host, s.info.User, s.info.Repository)
	fmt.Printf("  index path:                %s\n", s.indexPath)
	fmt.Printf("  TOP_K:                     %d\n", s.cfg.TopK)
	fmt.Printf("  OPENROUTER_MODEL:          %s\n", s.cfg.LLMModel)
}

func runAsk(s askSettings, question string) error {
	ctx := context.Background()
	p, err := pipeline.New(ctx, s.indexPath)
	if err != nil {
		log.Printf("ERROR ask: %v", err)
		return err
	}
	defer p.Close()

	runOne := func(q string) error {
		result, err := p.Ask(ctx, q)
		if err != nil {
			return err
		}

		fmt.Println("\nAnswer:")
		fmt.Println()
		fmt.Println(result.Answer)
		if result.RateLimited {
			fmt.Println("\n(See README: OPENROUTER_RAG_MODEL / rate limits.)")
			return nil
		}

		fmt.Println("\nSources (paths from retrieved chunks):")
		for _, source := range result.Sources {
			fmt.Printf("  - %s\n", source)
		}
		fmt.Printf("\n(retrieved_chunks=%d)\n", result.RetrievedChunks)
		return nil
	}

	question = strings.TrimSpace(question)
	if question != "" {
		if err := runOne(question); err != nil {
			log.Printf("ERROR ask: %v", err)
			return err
		}
		return nil
	}

	fmt.Println("RAG Q&A (type 'exit' or 'quit' to leave)")
	fmt.Println()
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("Question: ")
		if !scanner.Scan() {
			break
		}

		q := strings.TrimSpace(scanner.Text())
		if q == "" {
			continue
		}
		switch strings.ToLower(q) {
		case "exit", "quit", "q":
			fmt.Println("Bye!")
			return nil
		}

		if err := runOne(q); err != nil {
			log.Printf("ERROR ask: %v", err)
			return err
		}
		fmt.Println()
	}

	if err := scanner.Err(); err != nil {
		log.Printf("ERROR ask: %v", err)
		return err
	}
	return nil
}
