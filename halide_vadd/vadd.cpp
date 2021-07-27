#include "Halide.h"

using namespace Halide;

class VAdd : public Generator<VAdd> {
public:
    Input<Buffer<float>> a{"a", 1};
    Input<Buffer<float>> b{"b", 1};
    Output<Buffer<float>> c{"c", 1};

    void generate() {
        Var x;
        c(x) = a(x) + b(x);
    }
};

HALIDE_REGISTER_GENERATOR(VAdd, vadd);
