
HALIDE_PATH ?= /usr/local/halide
HALIDE_TOOL = $(HALIDE_PATH)/share/Halide/tools

.PHONY: run clean generate

run: vadd.rungen
	./$<

vadd.generator: vadd.cpp $(HALIDE_TOOL)/GenGen.cpp
	clang++-11 $^ -std=c++17 -I$(HALIDE_PATH)/include -L$(HALIDE_PATH)/lib -lHalide -Wl,-rpath,$(HALIDE_PATH)/lib -o $@

vadd.registration.cpp: generate

vadd.h: generate

vadd.a: generate

generate: vadd.generator
	./$< -g vadd -o . target=host

vadd.rungen: $(HALIDE_TOOL)/RunGenMain.cpp vadd.a vadd.registration.cpp
	clang++-11 -I$(HALIDE_PATH)/include $^ -o $@ -lpng -ljpeg -lz -lpthread -ldl

clean:
	rm -f *.rungen *.generator *.o *.out *.a *.h *.registration.cpp