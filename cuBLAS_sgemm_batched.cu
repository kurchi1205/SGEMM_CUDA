#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
 * A stand-alone script to invoke & benchmark standard cuBLAS SGEMM performance
 */

int main(int argc, char *argv[]) {
  int bs = 2
  int m = 2;
  int k = 3;
  int n = 4;
  int print = 1;
  cudaError_t cudaStat;  // cudaMalloc status
  cublasStatus_t stat;   // cuBLAS functions status
  cublasHandle_t handle; // cuBLAS context

  int i, j;

  float *a, *b, *c;

  // malloc for a,b,c...
  a = (float *)malloc(bs * m * k * sizeof(float));
  b = (float *)malloc(bs * k * n * sizeof(float));
  c = (float *)malloc(bs * m * n * sizeof(float));

  int ind = 11;
  for (j = 0; j < bs * m * k; j++) {
    a[j] = (float)ind++;
  }

  ind = 11;
  for (j = 0; j < bs * k * n; j++) {
    b[j] = (float)ind++;
  }

  ind = 11;
  for (j = 0; j < bs * m * n; j++) {
    c[j] = (float)ind++;
  }

  // DEVICE
  float *d_a, *d_b, *d_c;

  // cudaMalloc for d_a, d_b, d_c...
  cudaMalloc((void **)&d_a, bs * m * k * sizeof(float));
  cudaMalloc((void **)&d_b, bs * k * n * sizeof(float));
  cudaMalloc((void **)&d_c, bs * m * n * sizeof(float));

  stat = cublasCreate(&handle); // initialize CUBLAS context

  cudaMemcpy(d_a, a, bs * m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, bs * k * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, bs * m * n * sizeof(float), cudaMemcpyHostToDevice);

  float alpha = 1.0f;
  float beta = 0.5f;

  if (print == 1) {
    printf("alpha = %4.0f, beta = %4.0f\n", alpha, beta);
    printf("A = (bsxmxk: %d x %d x %d)\n", bs, m, k);
    for (batch = 0; batch < bs; batch++) {
      for (i = 0; i < m; i++) {
        for (j = 0; j < k; j++) {
          printf("%4.1f ", a[batch * bs + i * m + j]);
        }
        printf("\n");
      }
    }
    
    printf("B = (bsxkxn: %d x %d x %d)\n", bs, k, n);
    for (batch = 0; batch < bs; batch++) {
      for (i = 0; i < k; i++) {
        for (j = 0; j < n; j++) {
          printf("%4.1f ", b[batch * bs + i * n + j]);
        }
        printf("\n");
      }
    }

    printf("C = (bsxmxn: %d x %d x %d)\n", bs, m, n);
    for (batch = 0; batch < bs; batch++) {
      for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
          printf("%4.1f ", c[batch * bs + i * n + j]);
        }
        printf("\n");
      }
    }
  }

  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_b, n,
                     d_a, k, &beta, d_c, n);

  cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  if (print == 1) {
    printf("\nC after SGEMM = \n");
    for (batch = 0; batch < bs; batch++) {
      for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
          printf("%4.1f ", c[batch * bs + i * n + j]);
        }
        printf("\n");
      }
    }
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle); // destroy CUBLAS context
  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}