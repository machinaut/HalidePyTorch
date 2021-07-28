
HALIDE_PATH ?= /usr/local/halide
HALIDE_INC ?= $(HALIDE_PATH)/include
HALIDE_LIB ?= $(HALIDE_PATH)/lib

.PHONY: run clean

run: vadd
	./$<

vadd: vadd.cpp
	clang++-11 -std=c++17 $^ -I$(HALIDE_INC) -L$(HALIDE_LIB) -Wl,-rpath,$(HALIDE_LIB) -lHalide -lpthread -ldl -o $@

clean:
	rm -f vadd