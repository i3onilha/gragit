package ingestion

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/gen2brain/go-fitz"
	"golang.org/x/net/html"

	"github.com/i3onilha/gragit/internal/rag/document"
)

// skipDirNames are never descended into during directory ingest.
var skipDirNames = map[string]struct{}{
	".venv":        {},
	".git":         {},
	".cache":       {},
	".pixi":        {},
	"__pycache__":  {},
	"node_modules": {},
	"vendor":       {},
	"bin":          {},
	"dist":         {},
	"build":        {},
	"faiss_index":  {},
}

// binaryExtensions are skipped without attempting to read.
var binaryExtensions = map[string]struct{}{
	".so": {}, ".dll": {}, ".dylib": {}, ".exe": {}, ".bin": {}, ".o": {}, ".a": {},
	".pyc": {}, ".pyo": {}, ".class": {}, ".wasm": {},
	".png": {}, ".jpg": {}, ".jpeg": {}, ".gif": {}, ".webp": {}, ".ico": {}, ".bmp": {},
	".zip": {}, ".gz": {}, ".tar": {}, ".bz2": {}, ".7z": {}, ".rar": {}, ".xz": {},
	".pdf": {}, ".faiss": {}, ".pkl": {}, ".onnx": {}, ".pt": {}, ".pth": {}, ".safetensors": {},
	".woff": {}, ".woff2": {}, ".ttf": {}, ".eot": {}, ".otf": {},
	".mp3": {}, ".mp4": {}, ".avi": {}, ".mov": {}, ".mkv": {}, ".wav": {},
}

const maxTextBytes = 10 << 20 // 10 MiB

// LoadDocuments loads files or directories into documents with metadata["source"] set.
func LoadDocuments(paths []string) ([]document.Document, error) {
	files, err := expandPaths(paths)
	if err != nil {
		return nil, err
	}

	var docs []document.Document
	for _, path := range files {
		loaded, loadErr := loadFile(path)
		if loadErr != nil {
			log.Printf("WARNING ingestion: skipping unreadable file %s: %v", path, loadErr)
			continue
		}
		for _, doc := range loaded {
			meta := cloneMetadata(doc.Metadata)
			if _, ok := meta["source"]; !ok {
				meta["source"] = path
			}
			docs = append(docs, document.Document{
				PageContent: doc.PageContent,
				Metadata:    meta,
			})
		}
		log.Printf("INFO ingestion: loaded %s (%d segment(s))", path, len(loaded))
	}
	return docs, nil
}

func expandPaths(paths []string) ([]string, error) {
	var files []string
	for _, raw := range paths {
		path, err := filepath.Abs(filepath.Clean(raw))
		if err != nil {
			return nil, err
		}
		info, err := os.Stat(path)
		if err != nil {
			if os.IsNotExist(err) {
				log.Printf("WARNING ingestion: skipping missing path: %s", path)
				continue
			}
			return nil, err
		}
		if info.IsDir() {
			var found []string
			err = filepath.WalkDir(path, func(p string, d os.DirEntry, walkErr error) error {
				if walkErr != nil {
					return walkErr
				}
				if d.IsDir() {
					if p != path && shouldSkipDir(d.Name()) {
						return filepath.SkipDir
					}
					return nil
				}
				if shouldSkipFile(p) {
					return nil
				}
				found = append(found, p)
				return nil
			})
			if err != nil {
				return nil, err
			}
			if len(found) == 0 {
				log.Printf("WARNING ingestion: skipping empty directory: %s", path)
				continue
			}
			slices.Sort(found)
			files = append(files, found...)
			continue
		}
		if !shouldSkipFile(path) {
			files = append(files, path)
		}
	}
	return files, nil
}

func shouldSkipDir(name string) bool {
	_, skip := skipDirNames[name]
	return skip
}

func shouldSkipFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	if ext == ".pdf" || ext == ".html" || ext == ".htm" {
		return false
	}
	_, binary := binaryExtensions[ext]
	return binary
}

func loadFile(path string) ([]document.Document, error) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".pdf":
		return loadPDF(path)
	case ".html", ".htm":
		return loadHTML(path)
	default:
		return loadText(path)
	}
}

func loadText(path string) ([]document.Document, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if info.Size() > maxTextBytes {
		return nil, fmt.Errorf("file exceeds %d byte limit", maxTextBytes)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if !utf8.Valid(data) {
		return nil, fmt.Errorf("not valid UTF-8 text")
	}
	content := strings.TrimSpace(string(data))
	if content == "" {
		return nil, fmt.Errorf("empty text file")
	}
	return []document.Document{{
		PageContent: content,
		Metadata:    map[string]any{"source": path},
	}}, nil
}

func loadHTML(path string) ([]document.Document, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	node, err := html.Parse(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	var buf strings.Builder
	extractText(node, &buf)
	content := strings.TrimSpace(buf.String())
	if content == "" {
		return nil, fmt.Errorf("empty html file")
	}
	return []document.Document{{
		PageContent: content,
		Metadata:    map[string]any{"source": path},
	}}, nil
}

func extractText(n *html.Node, buf *strings.Builder) {
	if n.Type == html.TextNode {
		text := strings.TrimSpace(n.Data)
		if text != "" {
			if buf.Len() > 0 {
				buf.WriteByte(' ')
			}
			buf.WriteString(text)
		}
	}
	for child := n.FirstChild; child != nil; child = child.NextSibling {
		if child.Type == html.ElementNode {
			switch child.Data {
			case "script", "style", "noscript":
				continue
			}
		}
		extractText(child, buf)
	}
}

func loadPDF(path string) ([]document.Document, error) {
	doc, err := fitz.New(path)
	if err != nil {
		return nil, err
	}
	defer doc.Close()

	var docs []document.Document
	total := doc.NumPage()
	for pageNum := 0; pageNum < total; pageNum++ {
		text, err := doc.Text(pageNum)
		if err != nil {
			return nil, err
		}
		content := strings.TrimSpace(text)
		if content == "" {
			continue
		}
		docs = append(docs, document.Document{
			PageContent: content,
			Metadata: map[string]any{
				"source":      path,
				"page":        pageNum + 1,
				"total_pages": total,
			},
		})
	}
	if len(docs) == 0 {
		return nil, fmt.Errorf("no text extracted from pdf")
	}
	return docs, nil
}

func cloneMetadata(meta map[string]any) map[string]any {
	if meta == nil {
		return map[string]any{}
	}
	out := make(map[string]any, len(meta))
	for k, v := range meta {
		out[k] = v
	}
	return out
}
