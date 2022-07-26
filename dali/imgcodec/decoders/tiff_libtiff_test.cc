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

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "dali/imgcodec/decoders/tiff_libtiff.h"
#include "dali/imgcodec/parsers/tiff.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/mm/memory.h"
#include "dali/core/convert.h"
#include "dali/imgcodec/decoders/decoder_test_helper.h"

namespace dali {
namespace imgcodec {
namespace test {

const auto &dali_extra = dali::testing::dali_extra_path();
const std::string img_ref_path = dali_extra + "/db/single/reference/tiff/0/cat-111793_640.tiff.npy";
const std::string img_path = dali_extra + "/db/single/tiff/0/cat-111793_640.tiff";

class LibTiffDecoderTest : public NumpyDecoderTestBase<uint8_t> {
 protected:
  std::shared_ptr<ImageDecoderInstance> CreateDecoder(ThreadPool &tp) override {
    LibTiffDecoder decoder;
    return decoder.Create(CPU_ONLY_DEVICE_ID, tp);
  }
  std::shared_ptr<ImageParser> GetParser() override {
    return std::make_shared<TiffParser>();
  }
};

TEST_F(LibTiffDecoderTest, Test) {
  auto ref = ReadReferenceFrom(img_ref_path);
  auto src = ImageSource::FromFilename(img_path);
  auto img = Decode(&src);
  AssertEqual(img, ref);
}

TEST_F(LibTiffDecoderTest, TestROI) {
  auto ref = ReadReferenceFrom(img_ref_path);
  auto src = ImageSource::FromFilename(img_path);
  auto info = GetInfo(&src);

  DecodeParams params;
  params.use_roi = true;
  params.roi.begin = {13, 17, 0};
  params.roi.end = {info.shape[0] - 55, info.shape[1] - 10, 3};
  auto img = Decode(&src, params);
  AssertEqual(img, Crop(ref, params.roi.begin, params.roi.shape()));
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
