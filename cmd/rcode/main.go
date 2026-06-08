package main

import (
	"log"
	"os"

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
		Use:   "rcode",
		Short: "Index Git repositories into a local vector store",
	}

	root.AddCommand(newIngestCmd())
	root.AddCommand(newAskCmd())
	return root
}
