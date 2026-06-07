package gitrepo

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// Info describes the git repository detected from the working directory.
type Info struct {
	Host       string
	User       string
	Repository string
	Branch     string
	RemoteName string
	RemoteURL  string
	WorkDir    string
}

// IndexManifest records how and when an index was built.
type IndexManifest struct {
	Version           int    `json:"version"`
	RemoteURL         string `json:"remote_url"`
	Host              string `json:"host"`
	Owner             string `json:"owner"`
	Repository        string `json:"repository"`
	Branch            string `json:"branch"`
	CommitSHA         string `json:"commit_sha"`
	EmbeddingModel    string `json:"embedding_model"`
	ChunkSize         int    `json:"chunk_size"`
	ChunkOverlap      int    `json:"chunk_overlap"`
	IngestedAt        string `json:"ingested_at"`
	SourceClonePath   string `json:"source_clone_path"`
	VectorCount       int    `json:"vector_count"`
	EmbeddingDimension int   `json:"embedding_dimension"`
}

const manifestFile = "manifest.json"

// BaseDir returns the root directory for ragcode caches (~/.ragcode by default).
func BaseDir() (string, error) {
	if v := strings.TrimSpace(os.Getenv("GIT_FAISS_HOME")); v != "" {
		return filepath.Clean(v), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ragcode"), nil
}

// ModelsCacheDir returns ~/.ragcode/cache/models.
func ModelsCacheDir() (string, error) {
	base, err := BaseDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "cache", "models"), nil
}

// RepositoryDir returns ~/.ragcode/repos/<host>/<owner>/<repo>/<branch>.
func RepositoryDir(info Info) (string, error) {
	base, err := BaseDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "repos", info.Host, info.User, info.Repository, sanitizeBranch(info.Branch)), nil
}

// IndexDir returns ~/.ragcode/indexes/<host>/<owner>/<repo>/<branch>.
func IndexDir(info Info) (string, error) {
	base, err := BaseDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(
		base,
		"indexes",
		info.Host,
		info.User,
		info.Repository,
		sanitizeBranch(info.Branch),
	), nil
}

const defaultRemoteName = "origin"
const defaultBranch = "develop"

// ResolveFromCWD reads the configured remote and branch from the git repo containing cwd.
func ResolveFromCWD(remoteName, branch string) (Info, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return Info{}, err
	}
	return Resolve(cwd, remoteName, branch)
}

// Resolve reads the given remote and branch from the git repo containing workDir.
func Resolve(workDir, remoteName, branch string) (Info, error) {
	if strings.TrimSpace(remoteName) == "" {
		remoteName = defaultRemoteName
	}
	if strings.TrimSpace(branch) == "" {
		branch = defaultBranch
	}

	root, err := gitOutput(workDir, "rev-parse", "--show-toplevel")
	if err != nil {
		return Info{}, fmt.Errorf("not a git repository (run from inside a cloned repo): %w", err)
	}

	remoteURL, err := gitOutput(root, "remote", "get-url", remoteName)
	if err != nil {
		return Info{}, fmt.Errorf("no %s remote configured: %w", remoteName, err)
	}

	host, user, repo, err := parseRemoteURL(remoteURL)
	if err != nil {
		return Info{}, err
	}

	return Info{
		Host:       host,
		User:       user,
		Repository: repo,
		Branch:     strings.TrimSpace(branch),
		RemoteName: strings.TrimSpace(remoteName),
		RemoteURL:  strings.TrimSpace(remoteURL),
		WorkDir:    root,
	}, nil
}

// CommitSHA returns the full commit hash for HEAD in dir.
func CommitSHA(dir string) (string, error) {
	return gitOutput(dir, "rev-parse", "HEAD")
}

// Sync clones the remote repository or refreshes an existing clone to match remotes/<remote>/<branch>.
func Sync(info Info) (string, error) {
	dest, err := RepositoryDir(info)
	if err != nil {
		return "", err
	}

	if isGitRepo(dest) {
		if err := refreshClone(dest, info.Branch); err != nil {
			return "", err
		}
		return dest, nil
	}

	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return "", err
	}
	if err := runGit("", "clone", "--branch", info.Branch, "--single-branch", info.RemoteURL, dest); err != nil {
		return "", fmt.Errorf("clone %s: %w", info.RemoteURL, err)
	}
	return dest, nil
}

// IndexBundleComplete reports whether manifest.json, vectors.bin, and docstore.json exist.
func IndexBundleComplete(indexPath string) bool {
	for _, name := range []string{manifestFile, "vectors.bin", "docstore.json"} {
		if _, err := os.Stat(filepath.Join(indexPath, name)); err != nil {
			return false
		}
	}
	return true
}

// ReadIndexManifest loads manifest.json from an index directory.
func ReadIndexManifest(indexPath string) (IndexManifest, error) {
	data, err := os.ReadFile(filepath.Join(indexPath, manifestFile))
	if err != nil {
		return IndexManifest{}, fmt.Errorf("read manifest: %w", err)
	}
	var manifest IndexManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return IndexManifest{}, fmt.Errorf("parse manifest: %w", err)
	}
	return manifest, nil
}

// IndexMatchesSettings reports whether an existing manifest matches the given build parameters.
func IndexMatchesSettings(manifest IndexManifest, commitSHA, embeddingModel string, chunkSize, chunkOverlap int) bool {
	return manifest.CommitSHA == commitSHA &&
		manifest.EmbeddingModel == embeddingModel &&
		manifest.ChunkSize == chunkSize &&
		manifest.ChunkOverlap == chunkOverlap
}

// WriteIndexManifest persists manifest.json inside an index directory.
func WriteIndexManifest(indexPath string, manifest IndexManifest) error {
	manifest.Version = 1
	if manifest.IngestedAt == "" {
		manifest.IngestedAt = time.Now().UTC().Format(time.RFC3339)
	}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(indexPath, manifestFile), data, 0o644)
}

func refreshClone(dest, branch string) error {
	// Synced clones are created with `git clone` and always name the remote "origin".
	if err := runGit(dest, "fetch", defaultRemoteName); err != nil {
		return fmt.Errorf("fetch %s: %w", defaultRemoteName, err)
	}
	if err := runGit(dest, "checkout", branch); err != nil {
		return fmt.Errorf("checkout %s: %w", branch, err)
	}
	ref := defaultRemoteName + "/" + branch
	if err := runGit(dest, "reset", "--hard", ref); err != nil {
		return fmt.Errorf("reset --hard %s: %w", ref, err)
	}
	return nil
}

func parseRemoteURL(raw string) (host, user, repo string, err error) {
	raw = strings.TrimSpace(raw)
	raw = strings.TrimSuffix(raw, ".git")

	if strings.HasPrefix(raw, "git@") {
		rest := strings.TrimPrefix(raw, "git@")
		parts := strings.SplitN(rest, ":", 2)
		if len(parts) != 2 {
			return "", "", "", fmt.Errorf("invalid ssh git remote: %q", raw)
		}
		host = parts[0]
		user, repo, err = splitOwnerRepo(parts[1])
		return host, user, repo, err
	}

	u, parseErr := url.Parse(raw)
	if parseErr != nil {
		return "", "", "", fmt.Errorf("invalid git remote url %q: %w", raw, parseErr)
	}
	host = u.Host
	path := strings.Trim(u.Path, "/")
	if path == "" {
		return "", "", "", fmt.Errorf("git remote url has no repository path: %q", raw)
	}
	user, repo, err = splitOwnerRepo(path)
	return host, user, repo, err
}

func splitOwnerRepo(path string) (user, repo string, err error) {
	segments := strings.Split(path, "/")
	if len(segments) < 2 {
		return "", "", fmt.Errorf("git remote url has no owner/repository: %q", path)
	}
	user = segments[len(segments)-2]
	repo = segments[len(segments)-1]
	if user == "" || repo == "" {
		return "", "", fmt.Errorf("git remote url has empty owner or repository: %q", path)
	}
	return user, repo, nil
}

func sanitizeBranch(branch string) string {
	return strings.ReplaceAll(branch, "/", "--")
}

func isGitRepo(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, ".git"))
	return err == nil
}

func gitOutput(dir string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func runGit(dir string, args ...string) error {
	cmd := exec.Command("git", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
