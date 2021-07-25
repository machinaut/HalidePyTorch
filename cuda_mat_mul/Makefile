#!/usr/bin/env make

MATRIX_SIZE ?= 1024

CUDA_SDK ?= /usr/local/cuda
HALIDE_SDK ?= /home/aray/Halide-12.0.1-x86-64-linux
HALIDE_TOOLS ?= $(HALIDE_SDK)/share/Halide/tools/

.PHONY: all test clean

all: runner

mat_mul.generator: mat_mul_generator.cpp $(HALIDE_TOOLS)/GenGen.cpp
	g++ -I $(HALIDE_SDK)/include -o $@ $^ -L $(HALIDE_SDK)/lib -lHalide -Wl,-rpath,$(HALIDE_SDK)/lib

mat_mul.a: mat_mul.generator
	./$< -g mat_mul -e static_library,c_header,registration,stmt -o . target=host-cuda-cuda_capability_70 size=$(MATRIX_SIZE)

runner: runner.cpp mat_mul.a
	g++ -I $(HALIDE_SDK)/include -I $(HALIDE_TOOLS) -I . -I $(CUDA_SDK)/include -Wall $^ -o $@ -L $(CUDA_SDK)/lib64 -lpthread -ldl -lcudart -lcublas

test: runner
	HL_CUDA_JIT_MAX_REGISTERS=256 ./$< $(MATRIX_SIZE)

clean:
	rm -rf runner mat_mul.a mat_mul.generator
