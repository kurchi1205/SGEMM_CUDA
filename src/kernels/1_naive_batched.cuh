#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void sgemm_naive_batched(int Bs, int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int b = 0; b < Bs; ++b) {
      float tmp = 0.0;
      for (int i = 0; i < K; ++i) {
        tmp += A[b * M * K + x * K + i] * B[b * N * K + i * N + y];
      }
        // C = α*(A@B)+β*C
        C[b * M * N + x * N + y] = alpha * tmp + beta * C[b * M * N + x * N + y];
    }
  }
}

__global__ void sgemm_naive_batched_v2(int Bs, int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint b = blockIdx.z;
  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N && b < Bs) {
    float tmp = 0.0;
      for (int i = 0; i < K; ++i) {
        tmp += A[b * M * K + x * K + i] * B[b * N * K + i * N + y];
      }
        // C = α*(A@B)+β*C
     C[b * M * N + x * N + y] = alpha * tmp + beta * C[b * M * N + x * N + y];
  }
}