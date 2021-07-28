#include <stdio.h>
#include "HalideBuffer.h"
#include "vadd.h"

using namespace Halide;

void halide_register_argv_and_metadata(
    int (*filter_argv_call)(void **),
    const struct halide_filter_metadata_t *filter_metadata,
    const char *const *extra_key_value_pairs) {}

int main(void)
{
    Halide::Runtime::Buffer<float> a(123456789), b(123456789), c(123456789);

    for (int i = 0; i < 10; i++) {
        int error = vadd(a, b, c);

        if (error)
        {
            printf("Halide returned an error: %d\n", error);
            return -1;
        }
    }

    printf("Success\n");
    return 0;
}
