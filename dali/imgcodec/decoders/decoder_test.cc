#include "dali/imgcodec/decoders/decoder_test.h"
#include "dali/pipeline/data/tensor.h"


namespace dali {
namespace imgcodec {
namespace test {

Tensor<CPUBackend> CpuDecoderTest::ReadNumpy(const std::string &path) {
    std::cerr << "Hello there" << std::endl;
    return {};
}

TEST_F(CpuDecoderTest, TestXd) {
    ReadNumpy("/home/skarpinski/dummy-numpy");
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
