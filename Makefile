.PHONY: run-c run-go clean

all:	run-c run-go

run-c:	vectadd-c
	./vectadd-c

run-go:	vectadd-go
	./vectadd-go

vectadd-c:	vectadd.c
	gcc -std=c99 -o vectadd-c vectadd.c -lOpenCL -lm

vectadd-go:	vectadd.go
	go build -o vectadd-go vectadd.go

clean:
	rm -rf vectadd-c vectadd-go
