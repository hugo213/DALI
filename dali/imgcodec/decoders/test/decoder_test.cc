#include "dali/imgcodec/decoders/test/decoder_test.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/operators/reader/loader/numpy_loader.h"
#include "dali/kernels/transpose/transpose.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/imgcodec/decoders/test/numpy_helper.h"

namespace dali {
namespace imgcodec {
namespace test {

CpuDecoderTest::CpuDecoderTest() : tp_(4, CPU_ONLY_DEVICE_ID, false, "Decoder test") {}

void CpuDecoderTest::Compare(const std::string &image_path, const std::string &reference_path) {
  std::cerr << "CpuDecoderTest::Compare is a no-op for now!";
}

Tensor<CPUBackend> NumpyDecoderTest::DecodeReference(const std::string &reference_path) {
  return ReadNumpy(reference_path);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
