#!/usr/bin/env make

CUDA_PATH ?= /usr/local/cuda

.PHONY: run clean

run: vectorAdd
	./$<

vectorAdd.o: vectorAdd.cu
	nvcc -I $(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -arch=sm_70 -o $@ -c $<

vectorAdd.ptx: vectorAdd.cu
	nvcc -I $(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -arch=sm_70 -c $< --ptx

vectorAdd2.ptx: vectorAdd.o
	cuobjdump -ptx $< | c++filt > $@

vectorAdd.sass: vectorAdd.o
	cuobjdump -sass $< | c++filt > $@

vectorAdd.elf: vectorAdd.o
	cuobjdump -elf $< | c++filt > $@

vectorAdd.symbols: vectorAdd.o
	cuobjdump -symbols $< | c++filt > $@

vectorAdd.res-usage: vectorAdd.o
	cuobjdump -res-usage $< | c++filt > $@

vectorAdd: vectorAdd.o
	nvcc -o $@ $<

clean:
	rm -f vectorAdd *.o *.ptx *.sass *.elf *.symbols *.res-usage

