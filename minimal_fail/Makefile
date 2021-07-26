CUDA_PATH ?= /usr/local/cuda

.PHONY: run clean

run: zero.so
	CUDA_LAUNCH_BLOCKING=1 python zero.py

zero.so: zero.o
	nvcc -shared $^ -o $@ -lcuda

zero.o: zero.cu
	nvcc -I $(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc --compiler-options '-fPIC' $^ -c $@

clean:
	rm -f *.o *.so