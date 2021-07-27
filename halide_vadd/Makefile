
HALIDE_PATH ?= /usr/local/halide
HALIDE_TOOL = $(HALIDE_PATH)/share/Halide/tools

.PHONY: run clean

run: vadd_generator
	./$<

vadd_generator: vadd.cpp $(HALIDE_TOOL)/GenGen.cpp
	clang++-11 $^ -std=c++17 -I $(HALIDE_PATH)/include -L $(HALIDE_PATH)/lib -lHalide -Wl,-rpath,$(HALIDE_PATH)/lib -o $@

clean:
	rm -f vadd vadd_generator *.o *.out *.a