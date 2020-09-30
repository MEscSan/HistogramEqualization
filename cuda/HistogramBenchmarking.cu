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

void ColorConversionPageableBenchmarking(dim3 blocks, dim3 threadsPerBlock, int cpu_test)
{
    clock_t start_t, stop_t;
    
    
    double cpu_RGB2YCbCrTime = 0;
    double cuda_RGB2YCbCrTime  = 0; 
 
    for (int i = 0; i < 21; i++)
    {   
        cout.precision(5);
        string path = "../../Benchmark/"+ std::to_string(i) +".ppm";
        Image test(path.data());
        double miliseconds_CUDA = 0;
        double miliseconds_CPU = 0;

        // CUDA-Device
        miliseconds_CUDA += test.dev_rgb2yuv(blocks,threadsPerBlock);
        miliseconds_CUDA += test.dev_yuv2rgb(blocks,threadsPerBlock);
        cuda_RGB2YCbCrTime += miliseconds_CUDA;

        if(cpu_test)
        {
            // CPU
            start_t = clock();
            test.host_rgb2yuv();
            test.host_yuv2rgb();
            stop_t = clock();
            miliseconds_CPU +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
            cpu_RGB2YCbCrTime+= miliseconds_CPU; 
        }

    }

    cout <<'\t';
    cout <<cuda_RGB2YCbCrTime;
    cout <<"\t"<< cpu_RGB2YCbCrTime ;
    if( cpu_test)
    {
        double speedUp = cpu_RGB2YCbCrTime/cuda_RGB2YCbCrTime ;
        cout << '\t'<< speedUp;
    }
    cout << '\n';
}

void ColorConversionPinnedBenchmarking(dim3 blocks, dim3 threadsPerBlock, int cpu_test)
{
    clock_t start_t, stop_t;
    
    double cpu_RGB2YCbCrTime = 0;
    double cuda_RGB2YCbCrTime  = 0; 
 

    for (int i = 0; i < 21; i++)
    {   
        cout.precision(5);
        string path = "../../Benchmark/"+ std::to_string(i) +".ppm";
        Image test(path.data());
        double miliseconds_CUDA = 0;
        double miliseconds_CPU = 0;

        // CUDA-Device
        miliseconds_CUDA += test.dev_rgb2yuv_pinned(blocks,threadsPerBlock);
        miliseconds_CUDA += test.dev_yuv2rgb_pinned(blocks,threadsPerBlock);
        cuda_RGB2YCbCrTime += miliseconds_CUDA;

        if(cpu_test)
        {
            // CPU
            start_t = clock();
            test.host_rgb2yuv();
            test.host_yuv2rgb();
            stop_t = clock();
            miliseconds_CPU +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
            cpu_RGB2YCbCrTime+= miliseconds_CPU; 
        }

    }

    cout <<'\t';
    cout <<cuda_RGB2YCbCrTime;
    cout <<"\t"<< cpu_RGB2YCbCrTime ;
    if( cpu_test)
    {
        double speedUp = cpu_RGB2YCbCrTime/cuda_RGB2YCbCrTime ;
        cout << '\t'<< speedUp;
    }
    cout << '\n';
}

void ColorConversionUnifiedBenchmarking(dim3 blocks, dim3 threadsPerBlock, int cpu_test)
{
    clock_t start_t, stop_t;
    
    double cpu_RGB2YCbCrTime = 0;
    double cuda_RGB2YCbCrTime  = 0; 
 
    for (int i = 0; i < 21; i++)
    {   
        cout.precision(5);
        string path = "../../Benchmark/"+ std::to_string(i) +".ppm";
        Image test(path.data());
        double miliseconds_CUDA = 0;
        double miliseconds_CPU = 0;

        // CUDA-Device
        miliseconds_CUDA += test.dev_rgb2yuv_unified(blocks,threadsPerBlock);
        miliseconds_CUDA += test.dev_yuv2rgb_unified(blocks,threadsPerBlock);
        cuda_RGB2YCbCrTime += miliseconds_CUDA;

        if(cpu_test)
        {
            // CPU
            start_t = clock();
            test.host_rgb2yuv();
            test.host_yuv2rgb();
            stop_t = clock();
            miliseconds_CPU +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
            cpu_RGB2YCbCrTime+= miliseconds_CPU; 
        }

    }

    cout <<'\t';
    cout <<cuda_RGB2YCbCrTime;
    cout <<"\t"<< cpu_RGB2YCbCrTime ;
    if( cpu_test)
    {
        double speedUp = cpu_RGB2YCbCrTime/cuda_RGB2YCbCrTime ;
        cout << '\t'<< speedUp;
    }
    cout << '\n';
}

void RGB_HistogramOperationsBenchmarking(dim3 blocks, dim3 threadsPerBlock,int  cpu_test)
{
    clock_t start_t, stop_t;
    
    double cpu_histogramTime = 0;
    double cuda_histogramTime = 0; 


    for (int i = 0; i < 21; i++)
    {   
        cout.precision(5);
        string path = "../../Benchmark/"+ std::to_string(i) +".ppm";
        Image test(path.data());
        Histogram hist(test);
        double miliseconds_CUDA = 0;
	    double miliseconds_CPU = 0;

        // CUDA-Device
        miliseconds_CUDA += hist.dev_normalize(blocks,threadsPerBlock);
        miliseconds_CUDA += hist.dev_equalize(blocks,threadsPerBlock);
        cuda_histogramTime+= miliseconds_CUDA;

        if(cpu_test)
        {
            // CPU
            start_t = clock();
            hist.host_normalize();
            hist.host_equalize();
            stop_t = clock();
            miliseconds_CPU +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
            cpu_histogramTime+= miliseconds_CPU;
        }
    }

    
    cout <<'\t';
    cout <<cuda_histogramTime;
    cout <<"\t"<< cpu_histogramTime ;
    if(cpu_test)
    {
        double speedUp = cpu_histogramTime/cuda_histogramTime;
        cout << '\t'<< speedUp;
    }
    cout << '\n';
    
}

void GVP_HistogramOperationsBenchmarking(dim3 blocks, dim3 threadsPerBlock, int cpu_test)
{
    clock_t start_t, stop_t;
    
    double cpu_histogramTime = 0;
    double cuda_histogramTime = 0; 

    for (int i = 0; i < 21; i++)
    {   
        cout.precision(5);
        string path = "../../Benchmark/"+ std::to_string(i) +".pgm";
        Image test(path.data());
        Histogram hist(test);
        double miliseconds_CUDA = 0;
        double miliseconds_CPU = 0;

        // CUDA-Device
        miliseconds_CUDA += hist.dev_normalize(blocks,threadsPerBlock);
        miliseconds_CUDA += hist.dev_equalize(blocks,threadsPerBlock);
        cuda_histogramTime+= miliseconds_CUDA;

        if(cpu_test)
        {
            // CPU
            start_t = clock();
            hist.host_normalize();
            hist.host_equalize();
            stop_t = clock();
            miliseconds_CPU +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
            cpu_histogramTime+= miliseconds_CPU;
        }

    }

    
    cout <<'\t';
    cout <<cuda_histogramTime;
    cout <<"\t"<< cpu_histogramTime ;
    if(cpu_test)
    {
        double speedUp = cpu_histogramTime/cuda_histogramTime;
        cout << '\t'<< speedUp;
    }
    cout << '\n';
}

int main(int argc, char* argv[])
{

    //Over the command line the user can decide whether or not to benchmark against the CPUS
    int cpu_test = 0;

    if(argc > 1 )
    {
     	cpu_test = 1;
    }


    cudaDeviceProp deviceProperties;
    gpuErrchk(cudaGetDeviceProperties(&deviceProperties, 0));

    int numSM = deviceProperties.multiProcessorCount;
    int maxThreadsPerSM = deviceProperties.maxThreadsPerMultiProcessor;   

    cout<< "\nDevice: " << deviceProperties.name;
    cout<< "\nCompute Capability:\t\t" << deviceProperties.major, deviceProperties.minor;
    cout<< "\nClock Rate:\t\t\t" << deviceProperties.clockRate/1000 << " Hz";
    cout<< "\nNumber of SMs to be used:\t" << numSM;
    cout<< "\nMax Threads per SM:\t\t" << maxThreadsPerSM;
    cout<< "\nShared Memory per Block:\t" << deviceProperties.sharedMemPerBlock/1024 << " kB";
    cout<< "\nShared Memory per SM:\t\t" << deviceProperties.sharedMemPerMultiprocessor/1024 << " kB";
    cout<< "\nTotal Global Memory:\t\t" << deviceProperties.totalGlobalMem/1024/1024/1024 << " GB";
    
    cout<<"\n\nColor Conversion using Unified memory\n";
    
    cout<<"Blocks \tThreads\t";
    cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";     
    for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
    {
        int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
        cout<< blocks<<'\t' << threadsPerBlock;
        ColorConversionUnifiedBenchmarking(blocks, threadsPerBlock, cpu_test);
    }
    
    cout<<"\n\nColor Conversion using pageable memory\n";
    
    cout<<"Blocks \tThreads\t";
    cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";     
    for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
    {
        int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
        cout<< blocks<<'\t' << threadsPerBlock;
        ColorConversionPageableBenchmarking(blocks, threadsPerBlock, cpu_test);
    }

    cout<<"\nColor Conversion using pinned memory\n";
    
    cout<<"Blocks \tThreads\t";
    cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";     
    for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
    {
        int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
        cout<< blocks<<'\t' << threadsPerBlock;
        ColorConversionPinnedBenchmarking(blocks, threadsPerBlock, cpu_test);
    }

    cout<<"\nHistograms on grey value images\n";
    
    cout<<"Blocks \tThreads\t";
    cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";   
    for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
    {
        int blocks = numSM*maxThreadsPerSM/threadsPerBlock;

        cout<< blocks<<'\t' << threadsPerBlock;
        GVP_HistogramOperationsBenchmarking(blocks, threadsPerBlock, cpu_test);     
    }

    cout<<"\nHistograms on RGB-images\n";
    
    cout<<"Blocks \tThreads\t";
    cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";   
    for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
    {
        int blocks = numSM*maxThreadsPerSM/threadsPerBlock;

        cout<< blocks<<'\t' << threadsPerBlock;    
        RGB_HistogramOperationsBenchmarking(blocks, threadsPerBlock, cpu_test);
    }
    
    return EXIT_SUCCESS;
}
