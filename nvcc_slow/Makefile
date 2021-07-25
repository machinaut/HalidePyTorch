#!/usr/bin/env make

CUDA_PATH ?= /usr/local/cuda

# Old run

.PHONY: run clean

run: vectorAdd
	./$<

vectorAdd: vectorAdd.cu
	nvcc -I $(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -arch=sm_70 -o $@ $<

clean:
	rm -f vectorAdd *.o *.ptx *.sass *.elf *.symbols *.res-usage *.cpp.ii *.cubin *.fatbin

# New run

vectorAdd.cu.cpp.ii: vectorAdd.cu
	nvcc -I $(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -arch=sm_70 $< -cuda

# This should generate exact same cubin as below from the ptx
# vectorAdd.cubin: vectorAdd.cu
# 	nvcc -I $(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -arch=sm_70 $< -cubin

vectorAdd.ptx: vectorAdd.cu
	nvcc -I $(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -arch=sm_70 $< -ptx

vectorAdd.cubin: vectorAdd.ptx
	nvcc -I $(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -arch=sm_70 $< -cubin

vectorAdd.fatbin: vectorAdd.cubin
	nvcc -I $(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -arch=sm_70 $< -fatbin
