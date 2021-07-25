#!/usr/bin/env make

CUDA_PATH ?= /usr/local/cuda

.PHONY: run clean

run: vectorAdd
	./$<

vectorAdd.o: vectorAdd.cu
	nvcc -I $(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -o $@ -c $<

vectorAdd: vectorAdd.o
	nvcc -o $@ $<

clean:
	rm -f vectorAdd vectorAdd.o
