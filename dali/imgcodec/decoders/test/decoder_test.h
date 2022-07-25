// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_IMGCODEC_DECODERS_DECODER_TEST_H_
#define DALI_IMGCODEC_DECODERS_DECODER_TEST_H_

#include "dali/imgcodec/decoders/test/decoder_test.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/operators/reader/loader/numpy_loader.h"
#include "dali/kernels/transpose/transpose.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/imgcodec/decoders/test/numpy_helper.h"
#include "dali/imgcodec/image_format.h"
#include "dali/imgcodec/image_decoder.h"
#include "dali/test/dali_test.h"

namespace dali {
namespace imgcodec {
namespace test {

template<typename OutputType>
class CpuDecoderTest : public ::testing::Test {
 public:
  CpuDecoderTest() : tp_(4, CPU_ONLY_DEVICE_ID, false, "Decoder test") {}

  Tensor<CPUBackend> Decode(ImageSource *src) {
    if (!parser_) parser_ = GetParser();
    if (!decoder_) decoder_ = CreateDecoder(tp_);

    EXPECT_TRUE(parser_->CanParse(src));
    ImageInfo info = parser_->GetInfo(src);

    Tensor<CPUBackend> result;
    EXPECT_TRUE(decoder_->CanDecode(src, {}));
    result.Resize(info.shape, type2id<OutputType>::value);
    SampleView<CPUBackend> view(result.raw_mutable_data(), result.shape(), result.type());
    DecodeResult decode_result = decoder_->Decode(view, src, {});
    EXPECT_TRUE(decode_result.success);
    return result;
  }

  void AssertEqual(const Tensor<CPUBackend> &img, const Tensor<CPUBackend> &ref) {
    EXPECT_EQ(img.shape(), ref.shape()) << "Different shapes";
    auto img_view = view<const OutputType>(img), ref_view = view<const OutputType>(ref);
    for (int i = 0; i < volume(img.shape()); i++) {
      // TODO(skarpinski) Pretty-print position on error
      EXPECT_EQ(img_view.data[i], ref_view.data[i]);
    }
  }

  virtual Tensor<CPUBackend> ReadReference(const std::string &reference_path) = 0;

 protected:
  virtual std::shared_ptr<ImageDecoderInstance> CreateDecoder(ThreadPool &tp) = 0;
  virtual std::shared_ptr<ImageParser> GetParser() = 0;

 private:
  std::shared_ptr<ImageDecoderInstance> decoder_ = nullptr;
  std::shared_ptr<ImageParser> parser_ = nullptr;
  ThreadPool tp_;
};


template<typename OutputType>
class NumpyDecoderTest : public CpuDecoderTest<OutputType> {
 private:
  template<typename InputType>
  Tensor<CPUBackend> Convert(const Tensor<CPUBackend> &in) {
    Tensor<CPUBackend> out;
    out.Resize(in.shape(), type2id<OutputType>::value);
    auto in_view = view<const InputType>(in), out_view = view<OutputType>(out);
    for (int i = 0; i < volume(in.shape()); i++) {
      out_view.data[i] = ConvertSatNorm<OutputType>(in_view.data[i]);
    }
    return out;
  }
 public:
  Tensor<CPUBackend> ReadReference(const std::string &reference_path) override {
    Tensor<CPUBackend> ref = ReadNumpy(reference_path);

    DALIDataType output_type = type2id<OutputType>::value;
    TYPE_SWITCH(ref.type(), type2id, InputType, NUMPY_ALLOWED_TYPES, (
      return Convert<InputType>(ref);
    ), DALI_FAIL(make_string("Unsupported numpy reference type: ", ref.type())));  // NOLINT
  }
};

}  // namespace test
}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_DECODER_TEST_H_
