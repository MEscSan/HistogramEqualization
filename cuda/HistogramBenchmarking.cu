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

// Times the CUDA-Implementation vs the CPU-Implementation
void ColorConversionPageableBenchmarking(dim3 blocks, dim3 threadsPerBlock)
{
    clock_t start_t, stop_t;
    
    //Number of bytes processed (1 Byte pro pixel pro Image-Channel)
    long N = 0;
    
    double cpu_RGB2YCbCrTime = 0;
    double cuda_RGB2YCbCrTime  = 0; 
 
    //cout << "Rows\tCols\tTime Cuda[ms]\tTime CPU[ms]\n";
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
        //cout << test.getRows() << '\t' << test.getCols() << '\t' << miliseconds << '\t';

        N += 3*test.getRows()*test.getCols();

        // CPU
        start_t = clock();
        test.host_rgb2yuv();
        test.host_yuv2rgb();
        stop_t = clock();
        miliseconds_CPU +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
        cpu_RGB2YCbCrTime+= miliseconds_CPU;
        
        //cout << '\t' << miliseconds <<'\n';

    }

    double speedUp = cpu_RGB2YCbCrTime/cuda_RGB2YCbCrTime ;
    
    cout <<'\t';
    cout <<cuda_RGB2YCbCrTime;
    cout <<"\t"<< cpu_RGB2YCbCrTime ;
    cout << '\t'<< speedUp;
    cout << '\n';
}

void ColorConversionPinnedBenchmarking(dim3 blocks, dim3 threadsPerBlock)
{
    clock_t start_t, stop_t;
    
    //Number of bytes processed (1 Byte pro pixel pro Image-Channel)
    long N = 0;
    
    double cpu_RGB2YCbCrTime = 0;
    double cuda_RGB2YCbCrTime  = 0; 
 
    //cout << "Rows\tCols\tTime Cuda[ms]\tTime CPU[ms]\n";
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
        //cout << test.getRows() << '\t' << test.getCols() << '\t' << miliseconds << '\t';

        N += 3*test.getRows()*test.getCols();

        // CPU
        start_t = clock();
        test.host_rgb2yuv();
        test.host_yuv2rgb();
        stop_t = clock();
        miliseconds_CPU +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
        cpu_RGB2YCbCrTime+= miliseconds_CPU;
        
        //cout << '\t' << miliseconds <<'\n';

    }

    double speedUp = cpu_RGB2YCbCrTime/cuda_RGB2YCbCrTime ;
    
    cout <<'\t';
    cout <<cuda_RGB2YCbCrTime;
    cout <<"\t"<< cpu_RGB2YCbCrTime ;
    cout << '\t'<< speedUp;
    cout << '\n';
}

void RGB_HistogramOperationsBenchmarking(dim3 blocks, dim3 threadsPerBlock)
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

        // CPU
        start_t = clock();
        hist.host_normalize();
        hist.host_equalize();
        stop_t = clock();
        miliseconds_CPU +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
        cpu_histogramTime+= miliseconds_CPU;

    }

    double speedUp = cpu_histogramTime/cuda_histogramTime;
    cout <<'\t';
    cout <<cuda_histogramTime;
    cout <<"\t"<< cpu_histogramTime ;
    cout << '\t'<< speedUp;
    cout << '\n';
    
}

void GVP_HistogramOperationsBenchmarking(dim3 blocks, dim3 threadsPerBlock)
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

        // CPU
        start_t = clock();
        hist.host_normalize();
        hist.host_equalize();
        stop_t = clock();
        miliseconds_CPU +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
        cpu_histogramTime+= miliseconds_CPU;

    }

    double speedUp = cpu_histogramTime/cuda_histogramTime;
    cout <<'\t';
    cout <<cuda_histogramTime;
    cout <<"\t"<< cpu_histogramTime ;
    cout << '\t'<< speedUp;
    cout << '\n';
}

int main(int argc, char* argv[])
{

    cudaDeviceProp deviceProperties;
    int numSM;
    int maxThreadsPerSM;   
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
    if(argc != 0 && (int)*argv[0] - 48<numSM)
    {
     	 numSM = 1;
    }

    cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

    cout<<"\nColor Conversion using pageable memory\n";
    
    cout<<"Blocks \tThreads\t";
    cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";     
    for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
    {
        int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
        cout<< blocks<<'\t' << threadsPerBlock;
        ColorConversionPageableBenchmarking(blocks, threadsPerBlock);
    }

    cout<<"\nColor Conversion using pinned memory\n";
    
    cout<<"Blocks \tThreads\t";
    cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";     
    for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
    {
        int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
        cout<< blocks<<'\t' << threadsPerBlock;
        ColorConversionPinnedBenchmarking(blocks, threadsPerBlock);
    }

    /*
    cout<<"\nHistograms on grey value images\n";
    
    cout<<"Blocks \tThreads\t";
    cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";   
    for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
    {
        int blocks = numSM*maxThreadsPerSM/threadsPerBlock;

        cout<< blocks<<'\t' << threadsPerBlock;
        GVP_HistogramOperationsBenchmarking(blocks, threadsPerBlock);     
    }

    cout<<"\nHistograms on RGB-images\n";
    
    cout<<"Blocks \tThreads\t";
    cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";   
    for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
    {
        int blocks = numSM*maxThreadsPerSM/threadsPerBlock;

        cout<< blocks<<'\t' << threadsPerBlock;    
        RGB_HistogramOperationsBenchmarking(blocks, threadsPerBlock);
    }
    */
    return EXIT_SUCCESS;
}
