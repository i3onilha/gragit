package chunking

import (
	"log"
	"regexp"
	"strings"

	"github.com/linka-ai/git-rag/internal/rag/document"
)

var splitRegexCache = map[string]*regexp.Regexp{}

// SplitDocuments chunks documents with overlap, matching LangChain RecursiveCharacterTextSplitter defaults.
func SplitDocuments(docs []document.Document, chunkSize, chunkOverlap int) []document.Document {
	if len(docs) == 0 {
		return nil
	}

	splitter := &recursiveSplitter{
		chunkSize:    chunkSize,
		chunkOverlap: chunkOverlap,
		separators:   []string{"\n\n", "\n", " ", ""},
	}

	var chunks []document.Document
	for _, doc := range docs {
		parts := splitter.splitText(doc.PageContent, splitter.separators)
		for _, part := range parts {
			if strings.TrimSpace(part) == "" {
				continue
			}
			meta := cloneMetadata(doc.Metadata)
			chunks = append(chunks, document.Document{
				PageContent: part,
				Metadata:    meta,
			})
		}
	}

	for i := range chunks {
		chunks[i].Metadata = cloneMetadata(chunks[i].Metadata)
		chunks[i].Metadata["chunk_index"] = i
	}

	log.Printf("INFO chunking: split %d document(s) into %d chunk(s)", len(docs), len(chunks))
	return chunks
}

type recursiveSplitter struct {
	chunkSize    int
	chunkOverlap int
	separators   []string
}

func (s *recursiveSplitter) splitText(text string, separators []string) []string {
	finalChunks := []string{}

	separator := separators[len(separators)-1]
	newSeparators := []string{}
	for i, sep := range separators {
		if sep == "" {
			separator = sep
			break
		}
		if strings.Contains(text, sep) {
			separator = sep
			newSeparators = separators[i+1:]
			break
		}
	}

	splits := splitWithRegex(text, regexp.QuoteMeta(separator), true)
	goodSplits := []string{}
	joinSep := ""
	if separator != "" {
		joinSep = separator
	}

	for _, piece := range splits {
		if len(piece) < s.chunkSize {
			goodSplits = append(goodSplits, piece)
			continue
		}
		if len(goodSplits) > 0 {
			finalChunks = append(finalChunks, s.mergeSplits(goodSplits, joinSep)...)
			goodSplits = nil
		}
		if len(newSeparators) == 0 {
			finalChunks = append(finalChunks, piece)
		} else {
			finalChunks = append(finalChunks, s.splitText(piece, newSeparators)...)
		}
	}
	if len(goodSplits) > 0 {
		finalChunks = append(finalChunks, s.mergeSplits(goodSplits, joinSep)...)
	}
	return finalChunks
}

func (s *recursiveSplitter) mergeSplits(splits []string, separator string) []string {
	separatorLen := len(separator)
	docs := []string{}
	current := []string{}
	total := 0

	for _, part := range splits {
		partLen := len(part)
		extra := 0
		if len(current) > 0 {
			extra = separatorLen
		}
		if total+partLen+extra > s.chunkSize {
			if len(current) > 0 {
				if doc := joinDocs(current, separator); doc != "" {
					docs = append(docs, doc)
				}
				for total > s.chunkOverlap || (total+partLen+extra > s.chunkSize && total > 0) {
					removeLen := len(current[0])
					if len(current) > 1 {
						removeLen += separatorLen
					}
					total -= removeLen
					current = current[1:]
				}
			}
		}
		current = append(current, part)
		if len(current) == 1 {
			total = partLen
		} else {
			total += partLen + separatorLen
		}
	}

	if doc := joinDocs(current, separator); doc != "" {
		docs = append(docs, doc)
	}
	return docs
}

func joinDocs(docs []string, separator string) string {
	if len(docs) == 0 {
		return ""
	}
	text := strings.Join(docs, separator)
	return strings.TrimSpace(text)
}

func splitWithRegex(text, separator string, keepSeparator bool) []string {
	if separator == "" {
		return splitRunes(text)
	}

	pattern := "(" + separator + ")"
	re := splitRegexCache[pattern]
	if re == nil {
		re = regexp.MustCompile(pattern)
		splitRegexCache[pattern] = re
	}

	parts := re.Split(text, -1)
	if !keepSeparator {
		return parts
	}

	matches := re.FindAllStringIndex(text, -1)
	if len(matches) == 0 {
		return parts
	}

	var out []string
	for i, part := range parts {
		if part != "" {
			out = append(out, part)
		}
		if i < len(matches) {
			sep := text[matches[i][0]:matches[i][1]]
			if len(out) == 0 {
				out = append(out, sep)
			} else {
				out[len(out)-1] += sep
			}
		}
	}
	return out
}

func splitRunes(text string) []string {
	runes := []rune(text)
	out := make([]string, len(runes))
	for i, r := range runes {
		out[i] = string(r)
	}
	return out
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
