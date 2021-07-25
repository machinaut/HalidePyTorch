#!/usr/bin/env make

HALIDE_T ?= /opt/homebrew/Cellar/halide/12.0.1/share/tools
HALIDE_I ?= /opt/homebrew/Cellar/halide/12.0.1/include
HALIDE_L ?= /opt/homebrew/Cellar/halide/12.0.1/lib
CFLAGS ?= -g -O0 -std=c++17

.PHONY: run clean

tutorial: tutorial.cpp $(HALIDE_T)/GenGen.cpp
	c++ $(CFLAGS) -I $(HALIDE_I) -L $(HALIDE_L) -lHalide -o $@ $^

run: tutorial
	DYLD_LIBRARY_PATH=$(HALIDE_L) ./$<

# // g++ lesson_15*.cpp <path/to/tools/halide_image_io.h>/GenGen.cpp -g -std=c++17 -fno-rtti -I <path/to/Halide.h> -L <path/to/libHalide.so> -lHalide -o lesson_15_generate
# // bash lesson_15_generators_usage.sh




# all: vadd.generator

# vadd.generator: GenGen.o vadd_generator.o
# 	c++ $(CFLAGS) -o $@ -isystem $(HALIDE_I) -Wl,-search_paths_first -Wl,-headerpad_max_install_names -Wl,-rpath,$(HALIDE_L) $(HALIDE_LIB) $< 

# vadd_generator.o: vadd_generator.cpp
# 	c++ $(CFLAGS) -o $@ -c -isystem $(HALIDE_I) $<

# GenGen.o: $(HALIDE_T)/GenGen.cpp
# 	c++ $(CFLAGS)  -o $@ -c -isystem $(HALIDE_I) $<

# clean:
# 	rm -f *.o *.a *.generator

# /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ \
#     -DHALIDE_ENABLE_RTTI -DHALIDE_WITH_EXCEPTIONS \
#     -isystem /Users/aray/code/halide-install/include \
#     -O3 -DNDEBUG -arch arm64 -isysroot \
#     /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX11.3.sdk \
#     -std=c++17 \
#     -MD \
#     -MT CMakeFiles/depthwise_separable_conv.generator.dir/Users/aray/code/halide-install/share/tools/GenGen.cpp.o \
#     -MF CMakeFiles/depthwise_separable_conv.generator.dir/Users/aray/code/halide-install/share/tools/GenGen.cpp.o.d \
#     -o CMakeFiles/depthwise_separable_conv.generator.dir/Users/aray/code/halide-install/share/tools/GenGen.cpp.o \
#     -c /Users/aray/code/halide-install/share/tools/GenGen.cpp


# # [2/13] Build x_generator.o
# /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ \
#     -DHALIDE_ENABLE_RTTI -DHALIDE_WITH_EXCEPTIONS \
#     -isystem /Users/aray/code/halide-install/include \
#     -O3 -DNDEBUG -arch arm64 \
#     -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX11.3.sdk \
#     -std=c++17 \
#     -MD \
#     -MT CMakeFiles/depthwise_separable_conv.generator.dir/depthwise_separable_conv_generator.cpp.o \
#     -MF CMakeFiles/depthwise_separable_conv.generator.dir/depthwise_separable_conv_generator.cpp.o.d \
#     -o CMakeFiles/depthwise_separable_conv.generator.dir/depthwise_separable_conv_generator.cpp.o \
#     -c /Users/aray/code/Halide/apps/depthwise_separable_conv/depthwise_separable_conv_generator.cpp

# # [3/13] # build x.generator
# /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ \
#     -O3 -DNDEBUG -arch arm64 -isysroot \
#     /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX11.3.sdk \
#     -Wl,-search_paths_first -Wl,-headerpad_max_install_names \
#     CMakeFiles/depthwise_separable_conv.generator.dir/depthwise_separable_conv_generator.cpp.o \
#     CMakeFiles/depthwise_separable_conv.generator.dir/Users/aray/code/halide-install/share/tools/GenGen.cpp.o \
#     -o depthwise_separable_conv.generator  -Wl,-rpath,/Users/aray/code/halide-install/lib  \
#     /Users/aray/code/halide-install/lib/libHalide.13.0.0.dylib

# # [4/13] # run x.generator to generate standalone x.runtime
# cd /Users/aray/code/Halide/apps/depthwise_separable_conv/build
# /Users/aray/code/Halide/apps/depthwise_separable_conv/build/depthwise_separable_conv.generator \
#     -r depthwise_separable_conv.runtime -o . -e object target=host

# $(BIN)/%/add_float32.a: $(GENERATOR_BIN)/add.generator
# 	@mkdir -p $(@D)
# 	@echo Producing CPU operator
# 	@$^ -g add \
# 		$(ADD_TYPES_F32) \
# 		-f add_float32 \
# 		-e static_library,c_header,pytorch_wrapper \
# 		-o $(@D) \
# 		target=$* \
# 		auto_schedule=false
