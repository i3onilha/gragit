package document

// Document is a loaded text segment with citation metadata.
type Document struct {
	PageContent string
	Metadata    map[string]any
}
