// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <benchmark/benchmark.h>
#include "dali/benchmark/operator_bench.h"
#include "dali/benchmark/dali_bench.h"

namespace dali {
static void NormalDistributionArgs(benchmark::internal::Benchmark *b) {
  for (int batch_size = 256; batch_size >= 1; batch_size /= 2) {
      b->Args({batch_size, 128});
      b->Args({batch_size, 512});
  }
}

static void NormalDistributionNonUniformArgs(benchmark::internal::Benchmark *b) {
  for (int batch_size = 256; batch_size >= 1; batch_size /= 2) {
      b->Args({batch_size, 2048});
      b->Args({batch_size, 16*4096});
  }
}

BENCHMARK_DEFINE_F(OperatorBench, NormalDistributionX)(benchmark::State& st) {
  const int batch_size = st.range(0);
  const int sample_dim = st.range(1);

  auto spec = OpSpec("NormalDistribution")
      .AddArg("max_batch_size", batch_size)
      .AddArg("num_threads", 1)
      .AddArg("device", "gpu")
      .AddArg("dtype", DALI_FLOAT)
      .AddArg("mean", 0.1f)
      .AddArg("stddev", 1.5f);

  spec.AddInput("data", "gpu");

  this->RunGPU<float>(st, spec, batch_size, sample_dim, sample_dim, 3);
}

BENCHMARK_DEFINE_F(OperatorBench, NormalDistribution_NonUniform)(benchmark::State& st) {
  const int batch_size = st.range(0);
  const int max_sample_vol = st.range(1);

  TensorListShape<3> shape(batch_size);
  int vol_log2 = 0;
  for (int i = 1; i < max_sample_vol; i *= 2) vol_log2++;
  assert(vol_log2 > 0);
  shape.set_tensor_shape(0, {1, 1, 1});
  for (int i = 1; i < batch_size; ++i) {
    auto prev = volume(shape[i - 1]);
    int size = i % (div_ceil(batch_size, vol_log2)) == 0 ? prev * 2 : prev;
    shape.set_tensor_shape(i, {1, 1, size});
  }

  auto spec = OpSpec("NormalDistribution")
      .AddArg("max_batch_size", batch_size)
      .AddArg("num_threads", 1)
      .AddArg("device", "gpu")
      .AddArg("dtype", DALI_FLOAT)
      .AddArg("mean", 0.1f)
      .AddArg("stddev", 1.5f)
      .AddInput("data", "gpu");

  this->RunGPU<float>(st, spec, batch_size, shape);
}

BENCHMARK_REGISTER_F(OperatorBench, NormalDistributionX)
->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(NormalDistributionArgs);

BENCHMARK_REGISTER_F(OperatorBench, NormalDistribution_NonUniform)
->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(NormalDistributionNonUniformArgs);


}  // namespace dali
