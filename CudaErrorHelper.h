#pragma once

#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline void checkCuda(cudaError_t result)
{
  if (result != cudaSuccess)
  {
    printf("%s: %s\n", cudaGetErrorName(result), cudaGetErrorString(result));
    exit(EXIT_FAILURE);
  }
}

inline void verboseCudaReset()
{
  cudaError_t result = cudaGetLastError();
  printf("%s: %s\n", cudaGetErrorName(result), cudaGetErrorString(result));
}

inline void checkLastCuda()
{
  cudaError_t result = cudaGetLastError();
  if (result != cudaSuccess)
  {
    printf("%s: %s\n", cudaGetErrorName(result), cudaGetErrorString(result));
    exit(EXIT_FAILURE);
  }
}

inline void verboseCuda()
{
  cudaError_t result = cudaPeekAtLastError();
  printf("%s: %s\n", cudaGetErrorName(result), cudaGetErrorString(result));
}
