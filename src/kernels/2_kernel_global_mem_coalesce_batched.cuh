#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce_batched(int Bs, int M, int N, int K, float alpha, const float **A, const float **B, float beta, float **C) {
  const int batchIdx = blockIdx.z;
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // if statement is necessary to make things work under tile quantization
  if (cRow < M && cCol < N && batchIdx < Bs) {
    const float *A_batch = A[batchIdx];
    const float *B_batch = B[batchIdx];
    float *C_batch = C[batchIdx];

    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A_batch[cRow * K + i] * B_batch[i * N + cCol];
    }
    C_batch[cRow * N + cCol] = alpha * tmp + beta * C_batch[cRow * N + cCol];
  }
}
