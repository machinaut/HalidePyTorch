#include "Halide.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace Halide;
using namespace std;

int main(int argc, char **argv)
{
    int size = 12345;
    vector<int> dims{size};
    Buffer<float> a(dims);
    Buffer<float> b(dims);

    for (int x = 0; x < size; x++)
    {
        a(x) = rand() / (float)RAND_MAX;
        b(x) = rand() / (float)RAND_MAX;
    }

    Halide::Func c;
    Halide::Var x;

    c(x) = a(x) + b(x);

    Halide::Buffer<float> output = c.realize(dims);

    for (int i = 0; i < size; i++)
    {
        if (output(i) != (a(i) + b(i)))
        {
            printf("output(%d) = %f instead of %f\n", i, output(i), (a(i) + b(i)));
            printf("At x = %d\n", i);
            return -1;
        }
    }

    printf("Success!\n");

    return 0;
}