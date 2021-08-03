CUDA_PATH ?= /usr/local/cuda

.PHONY: run clean

run: main
	./$<

main: main.cu
	nvcc -I $(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -arch=sm_86 --compiler-options '-fPIC' $^ -o $@

vadd.so: vadd.o
	nvcc -shared $^ -o $@ -lcuda

vadd.o: vadd.cu
	nvcc -DTHREADS=32 -DVECTORIZE=1 -DUNROLL=2 -I $(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -arch=sm_86 --compiler-options '-fPIC' $^ -c $@

clean:
	rm -f *.o *.so main vadd
