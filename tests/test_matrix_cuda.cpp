#include "catch_amalgamated.hpp"
#include "matrix_cuda.hpp"

TEST_CASE("MatrixCUDA scalar addition", "[cuda][scalar]")
{
#ifdef USE_CUDA
    MatrixCUDA<float> A(2, 2, 1.0f);
    auto B = A + 2.0f;
    std::vector<float> host(4);
    B.copyToCPU(host.data());
    REQUIRE(host[0] == Approx(3.0f));
#else
    SUCCEED("CUDA disabled — skipping test");
#endif
}
