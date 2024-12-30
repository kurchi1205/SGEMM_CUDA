#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Please select a kernel (range 0 - 13, 0 for NVIDIA cuBLAS)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 13) {
    std::cerr << "Please enter a valid kernel number (0-13)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

  // print some device info
  // CudaDeviceInfo();

  // Declare the handle, create the handle, cublasCreate will return a value of
  // type cublasStatus_t to determine whether the handle was created
  // successfully (the value is 0)
  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cuBLAS FLOPs ceiling is reached at 8192
  std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

  long m, n, k, max_size, bs;
  bs = 2;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

  float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr; // host matrices
  float *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices

  A = (float *)malloc(sizeof(float) * bs * max_size * max_size);
  B = (float *)malloc(sizeof(float) * bs * max_size * max_size);
  C = (float *)malloc(sizeof(float) * bs * max_size * max_size);
  C_ref = (float *)malloc(sizeof(float) * bs * max_size * max_size);

  randomize_matrix(A, max_size * max_size * bs);
  randomize_matrix(B, max_size * max_size * bs);
  randomize_matrix(C, max_size * max_size * bs);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * bs * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * bs * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * bs * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * bs * max_size * max_size));

  cudaCheck(cudaMemcpy(dA, A, sizeof(float) * bs * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(float) * bs * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * bs * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * bs * max_size * max_size,
                       cudaMemcpyHostToDevice));


  

  int repeat_times = 50;
  for (int size : SIZE) {
    m = n = k = size;

    std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;

    float **Aarray = new float *[bs];
    float **Barray = new float *[bs];
    float **Carray = new float *[bs];

    for (int i = 0; i < bs; ++i) {
        Aarray[i] = dA + i * m * k;
        Barray[i] = dB + i * k * n;
        Carray[i] = dC + i * m * n;
    }

    float **d_Aarray, **d_Barray, **d_Carray;
    cudaMalloc(&d_Aarray, bs * sizeof(float *));
    cudaMalloc(&d_Barray, bs * sizeof(float *));
    cudaMalloc(&d_Carray, bs * sizeof(float *));

    cudaMemcpy(d_Aarray, Aarray, bs * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Barray, Barray, bs * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Carray, Carray, bs * sizeof(float *), cudaMemcpyHostToDevice);

    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (kernel_num != 0) {
      run_kernel(0, bs, m, n, k, alpha, dA, dB, beta, dC_ref, d_Aarray, d_Barray, d_Carray,
                 handle); // cuBLAS
      run_kernel(kernel_num, bs, m, n, k, alpha, dA, dB, beta, dC, d_Aarray, d_Barray, d_Carray,
                 handle); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(C, dC, sizeof(float) * bs * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(float) * bs * m * n, cudaMemcpyDeviceToHost);
      
      if (!verify_matrix(C_ref, C, m * n)) {
        std::cout
            << "Failed to pass the correctness verification against NVIDIA "
               "cuBLAS."
            << std::endl;
        if (m <= 128) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A:\n";
          print_matrix_batched(A, bs, m, n, fs);
          fs << "B:\n";
          print_matrix_batched(B, bs, m, n, fs);
          fs << "C:\n";
          print_matrix_batched(C, bs, m, n, fs);
          fs << "Should:\n";
          print_matrix_batched(C_ref, bs, m, n, fs);
        }
        exit(EXIT_FAILURE);
      }
    }

    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      // We don't reset dC between runs to save time
      run_kernel(kernel_num, bs, m, n, k, alpha, dA, dB, beta, dC, d_Aarray, d_Barray, d_Carray, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    long flops = 2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m);
    fflush(stdout);
    // make dC and dC_ref equal again (we modified dC while calling our kernel
    // for benchmarking)
    cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * bs * m * n,
                         cudaMemcpyDeviceToDevice));
    cudaFree(d_Aarray);
    cudaFree(d_Barray);
    cudaFree(d_Carray);
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cublasDestroy(handle);

  return 0;
};