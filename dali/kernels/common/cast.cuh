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

#ifndef DALI_KERNELS_COMMON_CAST_CUH_
#define DALI_KERNELS_COMMON_CAST_CUH_

#include "dali/core/convert.h"
#include "dali/kernels/common/block_setup.h"

namespace dali {
namespace kernels {

struct CastSampleDesc {
  void *output;
  const void *input;
};

template <typename OType, typename IType>
__global__ void BatchedCastKernel(const CastSampleDesc *samples, const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  auto *out = static_cast<OType *>(sample.output);
  const auto *in = static_cast<const IType *>(sample.input);
  for (int x = threadIdx.x + block.start.x; x < block.end.x; x += blockDim.x) {
    out[x] = ConvertSat<OType>(in[x]);
  }
}

template<int MaxSamples>
struct ExperimentalCastKernelParams {
  CastSampleDesc samples[MaxSamples];
  int sample_sizes[MaxSamples];
  int first_block[MaxSamples]; // Id of first block assigned to each sample
};

/* This kernel doesn't need any memcopies, because all data needed (except for the samples themselves,
 * of course) is passed as parameters. We pass SampleDescs via ExperimentalCastKernelParams.samples.
 * We don't pass BlockDescs because they are redundant, instead we compute block info in the kernel
 * itself. */
template <typename OType, typename IType, int MaxSamples>
__global__ void ExperimentalCastKernel(int nsamples, int block_volume_scale,
                                       ExperimentalCastKernelParams<MaxSamples> params) {
  CastSampleDesc sample;
  int size;
  int block_offset;

  // We don't have sampleIdx, so we need to find our sample this way
  #pragma unroll
  for (int i = 0; i < MaxSamples; i++) {
    if (i < nsamples && params.first_block[i] <= blockIdx.x) {
      sample = params.samples[i];
      size = params.sample_sizes[i];
      block_offset = blockIdx.x - params.first_block[i];
    }
  }

  // We also don't have BlockDesc, so we need to calculate block metadata in the kernel
  int block_size = block_volume_scale * blockDim.x;
  int block_start = block_offset * block_size;
  int block_end = block_start + block_size;
  if (block_end > size) block_end = size;

  // Now it's like in classic BatchedCastKernel
  auto *out = static_cast<OType *>(sample.output);
  const auto *in = static_cast<const IType *>(sample.input);
  for (int x = threadIdx.x + block_start; x < block_end; x += blockDim.x) {
    out[x] = ConvertSat<OType>(in[x]);
  }
}

}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_COMMON_CAST_CUH_
