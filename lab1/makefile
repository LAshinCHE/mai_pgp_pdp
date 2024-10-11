NVCC = nvcc
NVCCFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
CPP = g++
CPPFLAGS = --std=c++11 -g
BIN = lab_1
OBJCPP = lab1.cpp
OBJ = lab_1.cu
NAMECPP = lab1cpu
OBJTEST = lab_1_test_cuda_operation.cu

all: build

build:
	$(NVCC) $(NVCCFLAGS) $(OBJ) -o $(BIN)

build-test:
	$(NVCC) $(NVCCFLAGS) $(OBJTEST) -o test

build-cpu:
	$(CPP) $(CPPFLAGS) $(OBJCPP) -o $(NAMECPP)

clean:
	rm -rf *.o lab1_cpu $(BIN)