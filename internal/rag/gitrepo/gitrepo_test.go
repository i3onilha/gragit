package gitrepo

import (
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

func TestSanitizeModel(t *testing.T) {
	raw := "sentence-transformers/all-MiniLM-L6-v2"
	if got := sanitizeModel(raw); got != "sentence-transformers--all-MiniLM-L6-v2" {
		t.Fatalf("sanitizeModel() = %q", got)
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
	got, err := IndexDir(info, "all-MiniLM-L6-v2")
	if err != nil {
		t.Fatalf("IndexDir() error = %v", err)
	}
	want := filepath.Join(base, "indexes", "github.com", "acme", "my-project", "feature--login", "all-MiniLM-L6-v2")
	if got != want {
		t.Fatalf("IndexDir() = %q, want %q", got, want)
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
