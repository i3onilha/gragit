BINARY   := rcode
CMD      := ./cmd/rcode
GO       := go

.PHONY: all build test test-race fmt vet tidy install clean help

all: build

build:
	$(GO) build -o $(BINARY) $(CMD)

test:
	$(GO) test ./...

test-race:
	$(GO) test -race ./...

fmt:
	$(GO) fmt ./...

vet:
	$(GO) vet ./...

tidy:
	$(GO) mod tidy

install:
	$(GO) install $(CMD)

clean:
	rm -f $(BINARY)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build      Build the $(BINARY) binary (default)"
	@echo "  test       Run all tests"
	@echo "  test-race  Run tests with the race detector"
	@echo "  fmt        Format Go source"
	@echo "  vet        Run go vet"
	@echo "  tidy       Tidy module dependencies"
	@echo "  install    Install $(BINARY) to \$$GOPATH/bin"
	@echo "  clean      Remove built binary"
	@echo "  help       Show this help"
