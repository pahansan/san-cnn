all:
	go run -ldflags="-s -w" -gcflags="-B" -trimpath cmd/main.go
