NVCC = nvcc
NVCCFLAGS = --std=c++11
CPP = g++
CPPFLAGS = --std=c++11 -g
BIN = a.out
OBJ = main.cu

all: build

build:
	$(NVCC) $(NVCCFLAGS) $(OBJ) -o $(BIN)

run:
	./$(BIN)

build_cpu:
	$(CPP) $(CPPFLAGS) main_cpu.cpp -o a_cpu.out

clean: