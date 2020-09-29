#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CudaErrorHelper.h"
#include "CudaImageTools.h"
#include "CudaHistogramTools.h"

using namespace std;

int main()
{
    clock_t start_t, stop_t;

    // Color Conversion Benchmarking
    // Test Images are converted from RGB to YCbCr and back to RGB
    cout << "Color Conversion Benchmarking:\n";    
    
    double cpu_rgb_yCbCr_t = 0;
    double cuda_rgb_yCbCr_t = 0; 
    //double cuda_rgb_hsv_t = 0;
 
    for (int i = 0; i < 21; i++)
    {   
        string path = "../../TestImages/Benchmark/"+ std::to_string(i) +".ppm";
        Image test(path.data());

        cuda_rgb_yCbCr_t += test.dev_rgb2yuv();
        cuda_rgb_yCbCr_t += test.dev_yuv2rgb();

    }

    cout <<"CUDA RGB <-> YCrCb Conversion: " << cuda_rgb_yCbCr_t << " ms\n";

    start_t = clock();

    for (int i = 0; i < 21; i++)
    {   
        string path = "../../TestImages/Benchmark/"+ std::to_string(i) +".ppm";
        Image test(path.data());
        
        start_t = clock();
        test.host_rgb2yuv();
        test.host_yuv2rgb();
        stop_t = clock();
        cpu_rgb_yCbCr_t +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
    }

    cout <<"CPU RGB <-> YCrCb Conversion: " << cpu_rgb_yCbCr_t << " ms\n";
    
    

    return EXIT_SUCCESS;
}