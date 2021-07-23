#include "Halide.h"

using namespace Halide;

class VAddGenerator : public Halide::Generator<VAddGenerator> {
public:
    Var x;
    Input<Buffer<float>> input_a{"input_a", 1};
    Input<Buffer<float>> input_b{"input_b", 1};
    Output<Buffer<float>> output{"output", 1};
    void generate() {
        output(x) = input_a(x) + input_b(x);
    }
};

HALIDE_REGISTER_GENERATOR(VAddGenerator, vadd_generator)