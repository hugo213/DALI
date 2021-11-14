#!/bin/bash
./build/dali/python/nvidia/dali/test/dali_benchmark.bin \
	--benchmark_out="$1" \
	--benchmark_out_format="$2" \
	--benchmark_filter="$3"
