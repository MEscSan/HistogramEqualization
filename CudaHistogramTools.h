#pragma once 

#include <cstdlib>
#include <cstdio>
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CudaImageTools.h"

using namespace std;

//TODO: Implement RGB-handling
class Histogram
{
    private:
        int _numValues;
        int* _host_values;
        int* _dev_values;
        double* _host_valuesCumulative;
        double* _dev_valuesCumulative;
        unsigned char* _host_lookUpTable;
        unsigned char* _dev_lookUpTable;
        unsigned char* _dev_pixels;
        unsigned char _minValue;
        unsigned char _maxValue;
        // In order for changes in source-image to be consequently applied, we need to pass the source by reference, and not by value
        // Otherwise we would be passing only the pixel information by reference, but not the image object itself
        Image& _src;

    public:
        Histogram(Image& src, int host=0);

        void calculate(dim3 blocks = 128, dim3 threadsPerBlock = 128);
        void host_calculate();
        
        void display(ostream& output = cout);    
        
        void equalize(dim3 blocks = 128, dim3 threadsPerBlock = 128);
        void host_equalize();
        
        int* getHistogramPtr();
        
        void normalize(dim3 blocks = 128, dim3 threadsPerBlock = 128);
        void host_normalize();

        void save(string path);

};

int getMax(int* arrayPtr, int arraySize, colorSpace cs = colorSpace::gvp);
unsigned char getMax(unsigned char* arrayPtr, int arraySize, colorSpace cs = colorSpace::gvp);

__device__ int dev_getMax(int* arrayPtr, int arraySize, colorSpace cs = colorSpace::gvp);
__device__ unsigned char dev_getMax(unsigned char* arrayPtr, int arraySize, colorSpace cs = colorSpace::gvp);

__global__ void dev_calculate(unsigned char* pixelPtr, int* values, double* valuesCumulative, unsigned char* lookUpTable,  int rows, int cols, colorSpace color);

// Implement a one-dimensional version of the local-histograms-kernel proposed in https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
__global__ void partialHistograms(unsigned char* pixelPtr, int* g_partialHistograms,int numValues, int rows, int cols, int channels);

// Implement partial-histograms-reduction kernel as proposed in https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
__global__ void globalHistogram(int* g_partialHistograms, int* histogram, int numValues, int numPartialHistograms);

// Algorithm and code proposed in "GPU Gems 3", chapter 39 (Parallel Prefix Sum (Scan) with CUDA) for parallelization of prefix sum
__global__ void partialCumulativeHistograms(int* values, int* g_partialCumulative, int* sums, int numValues, int n);
__global__ void globalCumulativeHistogram(int* g_partialCumulative, int* sums, double* _dev_valuesCumulative, int numValues, int n, int rows, int cols);

__global__ void dev_equalize();
__global__ void dev_normalize();
