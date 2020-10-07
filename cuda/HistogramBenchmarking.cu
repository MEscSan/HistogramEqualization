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

void printBenchmark(double cudaTime, int cpu_test, double cpuTime)
{

    cout <<'\t';
    cout <<cudaTime;
    cout <<"\t"<< cpuTime ;
    if( cpu_test)
    {
        double speedUp = cpuTime/cudaTime ;
        cout << '\t'<< speedUp;
    }
    cout << '\n';

}

double CPU_ColorConversionBenchmark(string path)
{
    clock_t start_t, stop_t;

    double cpu_RGB2YCbCrTime = 0;
    
    cout.precision(5);
    
    Image test(path.data());
    double miliseconds_CPU = 0;

    start_t = clock();
    test.host_rgb2yuv();
    test.host_yuv2rgb();
    stop_t = clock();
    miliseconds_CPU +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
    cpu_RGB2YCbCrTime+= miliseconds_CPU; 

    return cpu_RGB2YCbCrTime;

}

double GPU_ColorConversionPageableBenchmark(dim3 blocks, dim3 threadsPerBlock, string path)
{
    
    double cuda_RGB2YCbCrTime = 0;
       
    cout.precision(5);
    Image test(path.data());
    double miliseconds_CUDA = 0;
    miliseconds_CUDA += test.dev_rgb2yuv(blocks,threadsPerBlock);
    miliseconds_CUDA += test.dev_yuv2rgb(blocks,threadsPerBlock);
    cuda_RGB2YCbCrTime += miliseconds_CUDA;


    return cuda_RGB2YCbCrTime;
}

double GPU_ColorConversionPinnedBenchmark(dim3 blocks, dim3 threadsPerBlock, string path)
{
  
    double cuda_RGB2YCbCrTime  = 0; 

    cout.precision(5);
    
    Image test(path.data());
    double miliseconds_CUDA = 0;
    miliseconds_CUDA += test.dev_rgb2yuv_pinned(blocks,threadsPerBlock);
    miliseconds_CUDA += test.dev_yuv2rgb_pinned(blocks,threadsPerBlock);
    cuda_RGB2YCbCrTime += miliseconds_CUDA;

    return cuda_RGB2YCbCrTime;
}

double GPU_ColorConversionUnifiedBenchmark(dim3 blocks, dim3 threadsPerBlock, string path)
{
    
    double cuda_RGB2YCbCrTime  = 0; 
  
    cout.precision(5);
    Image test(path.data());
    double miliseconds_CUDA = 0;
    miliseconds_CUDA += test.dev_rgb2yuv_unified(blocks,threadsPerBlock);
    miliseconds_CUDA += test.dev_yuv2rgb_unified(blocks,threadsPerBlock);
    cuda_RGB2YCbCrTime += miliseconds_CUDA;

    return cuda_RGB2YCbCrTime;
}

double CPU_RGB_HistogramOperationsBenchmark(string path)
{
    clock_t start_t, stop_t;
    double cpu_histogramTime = 0;

    cout.precision(5);
    //string path = "../../Benchmark/"+ std::to_string(i) +".ppm";
    Image test(path.data());
    Histogram hist(test);
    double miliseconds_CPU = 0;

    start_t = clock();
    hist.host_normalize();
    hist.host_equalize();
    stop_t = clock();
    miliseconds_CPU +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
    cpu_histogramTime+= miliseconds_CPU;

    return cpu_histogramTime;
}

double GPU_RGB_HistogramOperationsBenchmark(dim3 blocks, dim3 threadsPerBlock, string path)
{
    
    double cuda_histogramTime = 0; 
       
    cout.precision(5);
    //string path = "../../Benchmark/"+ std::to_string(i) +".ppm";
    Image test(path.data());
    Histogram hist(test);
    double miliseconds_CUDA = 0;

    // CUDA-Device
    miliseconds_CUDA += hist.dev_normalize(blocks,threadsPerBlock);
    miliseconds_CUDA += hist.dev_equalize(blocks,threadsPerBlock);
    cuda_histogramTime+= miliseconds_CUDA;

    return cuda_histogramTime;
    
}

double CPU_GVP_HistogramOperationsBenchmark(string path)
{
    double cpu_histogramTime = 0;
    clock_t start_t, stop_t;
  
    cout.precision(5);
    //string path = "../../Benchmark/"+ std::to_string(i) +".pgm";
    Image test(path.data());
    Histogram hist(test);
    double miliseconds_CPU = 0;

    start_t = clock();
    hist.host_normalize();
    hist.host_equalize();
    stop_t = clock();
    miliseconds_CPU +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
    cpu_histogramTime+= miliseconds_CPU;

    return cpu_histogramTime;
}

double GPU_GVP_HistogramOperationsBenchmark(dim3 blocks, dim3 threadsPerBlock, string path)
{
    
    double cuda_histogramTime = 0; 
  
    cout.precision(5);
    //string path = "../../Benchmark/"+ std::to_string(i) +".pgm";
    Image test(path.data());
    Histogram hist(test);
    double miliseconds_CUDA = 0;

    miliseconds_CUDA += hist.dev_normalize(blocks,threadsPerBlock);
    miliseconds_CUDA += hist.dev_equalize(blocks,threadsPerBlock);
    cuda_histogramTime+= miliseconds_CUDA;

    return cuda_histogramTime;
}

int main(int argc, char* argv[])
{

    // Over the command line the user can decide whether or not to benchmark against the CPU
    // If the first input-parameter is "0", no algorithm will be run on the CPU
    int cpu_test = 0;

    // The user can give an image path for benchmarking
    // The default path is "../../Benchmark/[1-20].p[p/g]m", where 20 RGB and 20 grey-value images can be found for benchmarking
    string  inputImgPath = "../../Benchmark/" ;
    string inputImgExtension = "ppm_pgm";

    double cpu_GVP_histogramTime = 0;
    double cpu_RGB_histogramTime = 0;
    double cpu_RGB2YCbCrTime = 0;

    double cuda_GVP_histogramTime = 0 ;
    double cuda_RGB_histogramTime = 0;
    double cuda_RGB2YCbCrTime = 0;


    // The second input-parameter ist the path to the benchmarking image
    if( argc > 2)
    {
        inputImgPath = argv[2];
        
        // Get image format
        inputImgExtension = inputImgPath.substr(inputImgPath.length() - 3, 3); 
    }

    if(argc > 1 && atoi(argv[1])!= 0  )
    {
        cpu_test = 1;

        // If the input image is an RGB-Image
        if(inputImgExtension == "ppm")
        {
            cpu_RGB2YCbCrTime = CPU_ColorConversionBenchmark
        (inputImgPath); 
            cpu_RGB_histogramTime = CPU_RGB_HistogramOperationsBenchmark(inputImgPath);    
        }
        // If the input image is a grey-value-picture
        else if(inputImgExtension == "pgm")
        {
            cpu_GVP_histogramTime = CPU_GVP_HistogramOperationsBenchmark(inputImgPath);
        }
        // Default-Benchmark with both RGB and grey-value images
        else
        {
            for(int i = 0; i < 21; i++ )
            {
                string defaultPath = inputImgPath + std::to_string(i) +".ppm";
                cpu_RGB2YCbCrTime += CPU_ColorConversionBenchmark(defaultPath); 
                cpu_RGB_histogramTime += CPU_RGB_HistogramOperationsBenchmark(defaultPath);

                defaultPath = inputImgPath + std::to_string(i) +".pgm";
                cpu_GVP_histogramTime += CPU_GVP_HistogramOperationsBenchmark(defaultPath);
            }
        }
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
    
    // If the input image is an RGB-Image
    if(inputImgExtension == "ppm" )
    {

        cout<<"\n\nColor Conversion using unified memory\n";
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";     
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_RGB2YCbCrTime = GPU_ColorConversionUnifiedBenchmark(blocks, threadsPerBlock, inputImgPath);
            printBenchmark(cuda_RGB2YCbCrTime, cpu_test, cpu_RGB2YCbCrTime);
        }
        
        cout<<"\n\nColor Conversion using pageable memory\n";
        
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";     
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_RGB2YCbCrTime = GPU_ColorConversionPageableBenchmark(blocks, threadsPerBlock, inputImgPath);
            printBenchmark(cuda_RGB2YCbCrTime, cpu_test, cpu_RGB2YCbCrTime);
        }

        cout<<"\nColor Conversion using pinned memory\n";
        
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";     
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_RGB2YCbCrTime = GPU_ColorConversionPinnedBenchmark(blocks, threadsPerBlock, inputImgPath);
            printBenchmark(cuda_RGB2YCbCrTime, cpu_test, cpu_RGB2YCbCrTime);
        }

        cout<<"\nHistograms on RGB-images\n";
    
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";   
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;    
            cuda_RGB_histogramTime = GPU_RGB_HistogramOperationsBenchmark(blocks, threadsPerBlock, inputImgPath);
            printBenchmark(cuda_RGB_histogramTime, cpu_test, cpu_RGB_histogramTime);
        }

    }
    // If the input image is a grey-value-picture
    else if(inputImgExtension == "pgm")
    {
        cout<<"\n\nHistograms on grey value images\n";
        
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";   
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;

            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_GVP_histogramTime = GPU_GVP_HistogramOperationsBenchmark(blocks, threadsPerBlock, inputImgPath);     
            printBenchmark(cuda_GVP_histogramTime, cpu_test, cpu_GVP_histogramTime);
        }
    }
    // Default-Benchmark with both RGB and grey-value images
    else
    {
       
        string defaultPath;

        cout<<"\n\nColor Conversion using Unified memory\n";
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";     
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_RGB2YCbCrTime = 0;
            for(int i = 0; i < 21; i++ )
            {
                defaultPath = inputImgPath + std::to_string(i) +".ppm";
                cuda_RGB2YCbCrTime += GPU_ColorConversionUnifiedBenchmark(blocks, threadsPerBlock, defaultPath);
            }

            printBenchmark(cuda_RGB2YCbCrTime, cpu_test, cpu_RGB2YCbCrTime);
        }
        
        cout<<"\n\nColor Conversion using pageable memory\n";
        
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";     
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_RGB2YCbCrTime = 0;
            for(int i = 0; i < 21; i++ )
            {
                defaultPath = inputImgPath + std::to_string(i) +".ppm";
                cuda_RGB2YCbCrTime += GPU_ColorConversionPageableBenchmark(blocks, threadsPerBlock, defaultPath);
            }

            printBenchmark(cuda_RGB2YCbCrTime, cpu_test, cpu_RGB2YCbCrTime);
        }

        cout<<"\nColor Conversion using pinned memory\n";
        
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";     
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_RGB2YCbCrTime = 0;
            for(int i = 0; i < 21; i++ )
            {
                defaultPath = inputImgPath + std::to_string(i) +".ppm";
                cuda_RGB2YCbCrTime += GPU_ColorConversionPinnedBenchmark(blocks, threadsPerBlock, defaultPath);
            }
            printBenchmark(cuda_RGB2YCbCrTime, cpu_test, cpu_RGB2YCbCrTime);
        }

        cout<<"\nHistograms on RGB-images\n";
    
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";   
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_RGB_histogramTime = 0;    
            for(int i = 0; i < 21; i++ )
            {
                defaultPath = inputImgPath + std::to_string(i) +".ppm";
                cuda_RGB_histogramTime += GPU_RGB_HistogramOperationsBenchmark(blocks, threadsPerBlock, defaultPath);
            }
            printBenchmark(cuda_RGB_histogramTime, cpu_test, cpu_RGB_histogramTime);
        }

        
        cout<<"\nHistograms on grey value images\n";
    
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";   
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock; 
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_GVP_histogramTime = 0;
            for(int i = 0; i < 21; i++ )
            {
                defaultPath = inputImgPath + std::to_string(i) +".pgm";
                cuda_GVP_histogramTime += GPU_GVP_HistogramOperationsBenchmark(blocks, threadsPerBlock, defaultPath);     
            }
            printBenchmark(cuda_GVP_histogramTime, cpu_test, cpu_GVP_histogramTime);
        }
    }
    

    return EXIT_SUCCESS;
}
