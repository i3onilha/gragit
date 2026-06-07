package gitrepo

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseRemoteURL(t *testing.T) {
	tests := []struct {
		name     string
		raw      string
		wantHost string
		wantUser string
		wantRepo string
	}{
		{
			name:     "https github",
			raw:      "https://github.com/acme/my-project.git",
			wantHost: "github.com",
			wantUser: "acme",
			wantRepo: "my-project",
		},
		{
			name:     "ssh github",
			raw:      "git@github.com:acme/my-project.git",
			wantHost: "github.com",
			wantUser: "acme",
			wantRepo: "my-project",
		},
		{
			name:     "nested group gitlab",
			raw:      "https://gitlab.com/group/subgroup/service.git",
			wantHost: "gitlab.com",
			wantUser: "subgroup",
			wantRepo: "service",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			host, user, repo, err := parseRemoteURL(tt.raw)
			if err != nil {
				t.Fatalf("parseRemoteURL() error = %v", err)
			}
			if host != tt.wantHost || user != tt.wantUser || repo != tt.wantRepo {
				t.Fatalf("parseRemoteURL() = (%q, %q, %q), want (%q, %q, %q)",
					host, user, repo, tt.wantHost, tt.wantUser, tt.wantRepo)
			}
		})
	}
}

func TestSanitizeBranch(t *testing.T) {
	if got := sanitizeBranch("feature/login"); got != "feature--login" {
		t.Fatalf("sanitizeBranch() = %q, want feature--login", got)
	}
}

func TestIndexDirLayout(t *testing.T) {
	base := t.TempDir()
	t.Setenv("GIT_FAISS_HOME", base)

	info := Info{
		Host:       "github.com",
		User:       "acme",
		Repository: "my-project",
		Branch:     "feature/login",
	}
	got, err := IndexDir(info)
	if err != nil {
		t.Fatalf("IndexDir() error = %v", err)
	}
	want := filepath.Join(base, "indexes", "github.com", "acme", "my-project", "feature--login")
	if got != want {
		t.Fatalf("IndexDir() = %q, want %q", got, want)
	}
}

func TestIndexMatchesSettings(t *testing.T) {
	manifest := IndexManifest{
		CommitSHA:      "abc123",
		EmbeddingModel: "all-MiniLM-L6-v2",
		ChunkSize:      1000,
		ChunkOverlap:   200,
	}

	if !IndexMatchesSettings(manifest, "abc123", "all-MiniLM-L6-v2", 1000, 200) {
		t.Fatal("expected matching settings to be current")
	}
	if IndexMatchesSettings(manifest, "def456", "all-MiniLM-L6-v2", 1000, 200) {
		t.Fatal("expected different commit to be stale")
	}
	if IndexMatchesSettings(manifest, "abc123", "other-model", 1000, 200) {
		t.Fatal("expected different model to be stale")
	}
	if IndexMatchesSettings(manifest, "abc123", "all-MiniLM-L6-v2", 500, 200) {
		t.Fatal("expected different chunk size to be stale")
	}
}

func TestIndexBundleComplete(t *testing.T) {
	dir := t.TempDir()
	if IndexBundleComplete(dir) {
		t.Fatal("expected empty directory to be incomplete")
	}

	for _, name := range []string{"manifest.json", "vectors.bin", "docstore.json"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	if !IndexBundleComplete(dir) {
		t.Fatal("expected complete bundle to be detected")
	}
}

func TestReadWriteIndexManifest(t *testing.T) {
	dir := t.TempDir()
	want := IndexManifest{
		CommitSHA:      "abc123",
		EmbeddingModel: "all-MiniLM-L6-v2",
		ChunkSize:      1000,
		ChunkOverlap:   200,
		VectorCount:    42,
	}

	if err := WriteIndexManifest(dir, want); err != nil {
		t.Fatalf("WriteIndexManifest() error = %v", err)
	}

	got, err := ReadIndexManifest(dir)
	if err != nil {
		t.Fatalf("ReadIndexManifest() error = %v", err)
	}
	if got.CommitSHA != want.CommitSHA || got.VectorCount != want.VectorCount {
		t.Fatalf("ReadIndexManifest() = %+v, want commit %q vector_count %d", got, want.CommitSHA, want.VectorCount)
	}
}

func TestRepositoryDirLayout(t *testing.T) {
	base := t.TempDir()
	t.Setenv("GIT_FAISS_HOME", base)

	info := Info{
		Host:       "github.com",
		User:       "acme",
		Repository: "my-project",
		Branch:     "develop",
	}
	got, err := RepositoryDir(info)
	if err != nil {
		t.Fatalf("RepositoryDir() error = %v", err)
	}
	want := filepath.Join(base, "repos", "github.com", "acme", "my-project", "develop")
	if got != want {
		t.Fatalf("RepositoryDir() = %q, want %q", got, want)
	}
}
