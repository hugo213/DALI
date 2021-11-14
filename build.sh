#!/bin/bash

set -xe

# Zakomentowane, żeby nie kasować builda za każdym razem :P
# rm -rf build
# mkdir build

cd build


CUDA_VERSION=11.5 \
cmake -D CMAKE_BUILD_TYPE=Release \
-D BUILD_LMDB=OFF \
-D BUILD_LIBTIFF=OFF \
-D BUILD_LIBSND=OFF \
-D BUILD_LIBTAR=OFF \
-D BUILD_NVOF=OFF \
-D BUILD_CUFILE=OFF \
-D BUILD_BENCHMARK=ON \
-D CUDA_VERSION=11.5 \
-D NVJPEG_ROOT_DIR=/usr/local/cuda \
-D FFMPEG_ROOT_DIR=/usr/local/ \
..

# Flaga dla ffmpeg FFMEG -> zobacz https://github.com/NVIDIA/DALI/issues/2659

make -j20
