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

#include "dali/imgcodec/image_decoder.h"
#include "dali/test/dali_test.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/imgcodec/decoders/test/numpy_helper.h"

namespace dali {
namespace imgcodec {
namespace test {

class CpuDecoderTest : public ::testing::Test {
 public:
  CpuDecoderTest();

  void Compare(const std::string &image_path, const std::string &reference_path);

  Tensor<CPUBackend> Decode(const std::string &image);

  virtual Tensor<CPUBackend> DecodeReference(const std::string &reference_path) = 0;

 protected:
  virtual std::shared_ptr<ImageDecoderInstance> CreateDecoder(ThreadPool &tp) = 0;

 private:
  ThreadPool tp_;
  std::shared_ptr<ImageDecoderInstance> decoder_instance_;
};

class NumpyDecoderTest : public CpuDecoderTest {
 public:
  Tensor<CPUBackend> DecodeReference(const std::string &reference_path) override;
};

}  // namespace test
}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_DECODER_TEST_H_
