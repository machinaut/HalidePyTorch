
HALIDE_PATH ?= /usr/local/halide
HALIDE_TOOL = $(HALIDE_PATH)/share/Halide/tools

.PHONY: run benchmark clean generate

run: vadd
	./$<

vadd: vadd.cpp vadd.a vadd.registration.cpp
	clang++-11 -I$(HALIDE_PATH)/include $^ -o $@ -lpthread -ldl

benchmark: vadd.rungen
	./$< --benchmarks=all --benchmark_min_time=1 a=random:42:[123456789] b=random:42:[123456789] --parsable_output

vadd.generator: vadd.generator.cpp $(HALIDE_TOOL)/GenGen.cpp
	clang++-11 $^ -I$(HALIDE_PATH)/include -L$(HALIDE_PATH)/lib -lHalide -Wl,-rpath,$(HALIDE_PATH)/lib -o $@

vadd.registration.cpp: generate

vadd.h: generate

vadd.a: generate

generate: vadd.generator
	./$< -g vadd -o . target=host-cuda_capability_70

vadd.rungen: $(HALIDE_TOOL)/RunGenMain.cpp vadd.a vadd.registration.cpp
	clang++-11 -I$(HALIDE_PATH)/include $^ -o $@ -lpng -ljpeg -lz -lpthread -ldl

clean:
	rm -f *.rungen *.generator *.o *.out *.a *.h *.registration.cpp