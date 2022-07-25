#include "dali/imgcodec/decoders/test/decoder_test.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/operators/reader/loader/numpy_loader.h"
#include "dali/kernels/transpose/transpose.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/imgcodec/decoders/test/numpy_helper.h"

#include "dali/imgcodec/image_format.h"
#include "dali/imgcodec/image_decoder.h"

namespace dali {
namespace imgcodec {
namespace test {

CpuDecoderTest::CpuDecoderTest() : tp_(4, CPU_ONLY_DEVICE_ID, false, "Decoder test") {}

Tensor<CPUBackend> CpuDecoderTest::Decode(ImageSource *src) {
  if (!parser_) parser_ = GetParser();
  if (!decoder_) decoder_ = CreateDecoder(tp_);

  EXPECT_TRUE(parser_->CanParse(src));
  ImageInfo info = parser_->GetInfo(src);

  Tensor<CPUBackend> result;

  EXPECT_TRUE(decoder_->CanDecode(src, {}));

  return result;
}

Tensor<CPUBackend> NumpyDecoderTest::DecodeReference(const std::string &reference_path) {
  return ReadNumpy(reference_path);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
