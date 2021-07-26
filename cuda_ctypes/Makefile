CUDA_PATH ?= /usr/local/cuda

.PHONY: run clean

run: vadd.py vadd.so
	python $<

vadd.so: vadd.o
	nvcc -shared $^ -o $@ -lcuda

vadd.o: vadd.cu
	nvcc -I $(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -arch=sm_70 --compiler-options '-fPIC' $^ -c $@

clean:
	rm -f *.o *.so *.a