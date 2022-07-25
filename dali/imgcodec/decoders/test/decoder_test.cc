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

TEST_F(CpuDecoderTest, TestXd) {
  auto t = ReadNumpy("/home/skarpinski/DALI_extra/db/single/reference/tiff/0/cat-111793_640.tiff.npy");
  std::cerr << t.shape();
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
