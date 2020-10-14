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

// Equalizes and Normalizes becnhmark-images using the methods and classes implemented in the project and shows the execution time
// Users can decide whether to benchmark he sequential (cpu) version of the algorithm
// Users can give their own benchmark-image path
//
// Command line call:
// ./EqualizationTest [cpu_test] [inputImgPath]
// [cpu_test]: 
//      0 => algorithms run only on the gpu
//      else => algorithms run both on gpu(parallel) and cpu(sequential)
// [inputImgPath]: path to the test image
//
// Remark:  gvp...grey-value-picture (in code and comments) 

// Prints algorihtm execution time on the console
// cudaTime = parallel-algorithm execution time on CUDA device + device-to-host copy time + host-to-device copy time
// cpuTime = sequential-algorithm execution time on cpu
void printBenchmark(double cudaTime, int cpu_test, double cpuTime)
{
    // Print results with 5 decimal figures
    cout.precision(5);

    cout <<'\t';
    cout <<cudaTime;
    cout <<"\t"<< cpuTime ;
    // In execution time of sequential algorithms on the cpu is to be given, show it and get the speedup (sequential/parallel)
    if( cpu_test)
    {
        double speedUp = cpuTime/cudaTime ;
        cout << '\t'<< speedUp;
    }
    cout << '\n';

}

// Runs and times the execution  of sequential RGB-to YCbCr-Conversion on the cpu   
double CPU_ColorConversionBenchmark(string path)
{
    clock_t start_t, stop_t;

    double cpu_RGB2YCbCrTime = 0;

    // Initialize image object with input image
    Image test(path.data());

    // Sequential RGB-to-YCbCr and YCrCb-to-RGB conversion on the cpu
    start_t = clock();
    test.host_rgb2yuv();
    test.host_yuv2rgb();
    stop_t = clock();
    
    // Get the execution time in milliseconds
    cpu_RGB2YCbCrTime =1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC; 

    return cpu_RGB2YCbCrTime;

}

// Runs and times the execution time of parallel RGB-to YCbCr-Conversion on the CUDA device using pageable memory
double GPU_ColorConversionPageableBenchmark(dim3 blocks, dim3 threadsPerBlock, string path)
{
    
    double cuda_RGB2YCbCrTime = 0;
       
    // Initialize image object with input image
    Image test(path.data());

    // Paralllel RGB-to-YCbCr and YCrCb-to-RGB conversion on the CUDA device
    // Pageable memory is used
    cuda_RGB2YCbCrTime += test.dev_rgb2yuv(blocks,threadsPerBlock);
    cuda_RGB2YCbCrTime += test.dev_yuv2rgb(blocks,threadsPerBlock);

    return cuda_RGB2YCbCrTime;
}

// Runs and times the execution of parallel RGB-to YCbCr-Conversion on the CUDA device using pinned memory
double GPU_ColorConversionPinnedBenchmark(dim3 blocks, dim3 threadsPerBlock, string path)
{
    double cuda_RGB2YCbCrTime  = 0; 

    // Initialize image object with input image
    Image test(path.data());

    // Paralllel RGB-to-YCbCr and YCrCb-to-RGB conversion on the CUDA device
    // Pinned memory is used
    cuda_RGB2YCbCrTime += test.dev_rgb2yuv_pinned(blocks,threadsPerBlock);
    cuda_RGB2YCbCrTime += test.dev_yuv2rgb_pinned(blocks,threadsPerBlock);

    return cuda_RGB2YCbCrTime;
}

// Runs and times the execution of parallel RGB-to YCbCr-Conversion on the CUDA device using unified memory
double GPU_ColorConversionUnifiedBenchmark(dim3 blocks, dim3 threadsPerBlock, string path)
{
    double cuda_RGB2YCbCrTime  = 0; 

    // Initialize image object with input image
    Image test(path.data());

    // Paralllel RGB-to-YCbCr and YCrCb-to-RGB conversion on the CUDA device
    // Unified memory is used
    cuda_RGB2YCbCrTime += test.dev_rgb2yuv_unified(blocks,threadsPerBlock);
    cuda_RGB2YCbCrTime += test.dev_yuv2rgb_unified(blocks,threadsPerBlock);

    return cuda_RGB2YCbCrTime;
}

// Runs and times the execution of sequential histogram normalization and equalization of an RGB image on the cpu   
double CPU_RGB_HistogramOperationsBenchmark(string path)
{
    clock_t start_t, stop_t;
    double cpu_histogramTime = 0;

    // Initialize Image and Histogram objet from input
    Image test(path.data());
    Histogram hist(test);

    //Sequential histogram normalization and equalization on the cpu
    start_t = clock();
    hist.host_normalize();
    hist.host_equalize();
    stop_t = clock();

    // Get execution time in miliseconds
    cpu_histogramTime =1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;

    return cpu_histogramTime;
}

// Runs and times execution of parallel histogram normalization and equalization of an RGB image on CUDA device
double GPU_RGB_HistogramOperationsBenchmark(dim3 blocks, dim3 threadsPerBlock, string path)
{
    double cuda_histogramTime = 0; 
       
    // Initialize Image and Histogram objet from input
    Image test(path.data());
    Histogram hist(test);

    // normalize and equalize image and histogram on CUDA-Device
    cuda_histogramTime += hist.dev_normalize(blocks,threadsPerBlock);
    cuda_histogramTime += hist.dev_equalize(blocks,threadsPerBlock);

    return cuda_histogramTime;
    
}

// Runs and times execution of sequential histogram normalization and equalization of a gvp image on the cpu
double CPU_GVP_HistogramOperationsBenchmark(string path)
{
    double cpu_histogramTime = 0;
    clock_t start_t, stop_t;

    // Initialize Image and Histogram objet from input
    Image test(path.data());
    Histogram hist(test);

    // normalize and equalize image and histogram 
    start_t = clock();
    hist.host_normalize();
    hist.host_equalize();
    stop_t = clock();

    // Get execution time in milliseconds
    cpu_histogramTime+=  1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;

    return cpu_histogramTime;
}

// Runs and times execution of parallel histogram normalization and equalization of a gvp image on CUDA device
double GPU_GVP_HistogramOperationsBenchmark(dim3 blocks, dim3 threadsPerBlock, string path)
{  
    double cuda_histogramTime = 0; 

    // Initialize Image and Histogram objects from input
    Image test(path.data());
    Histogram hist(test);

    cuda_histogramTime += hist.dev_normalize(blocks,threadsPerBlock);
    cuda_histogramTime += hist.dev_equalize(blocks,threadsPerBlock);

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

    // Initialize benchmarking variables
    double cpu_GVP_histogramTime = 0;
    double cpu_RGB_histogramTime = 0;
    double cpu_RGB2YCbCrTime = 0;

    double cuda_GVP_histogramTime = 0 ;
    double cuda_RGB_histogramTime = 0;
    double cuda_RGB2YCbCrTime = 0;


    // The second input-parameter is the path to the benchmarking image
    if( argc > 2)
    {
        inputImgPath = argv[2];
        
        // Get image format
        inputImgExtension = inputImgPath.substr(inputImgPath.length() - 3, 3); 
    }

    // The first input-parameter turns the cpu calculation of the algorithms on and off(default)
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

    // Get and print main properties of the CUDA device
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
    
    // Run and time algorithms on CUDA device
    // If the input image is an RGB-Image
    if(inputImgExtension == "ppm" )
    {

        // RGB <-> YCbCr using unified memory
        cout<<"\n\nColor Conversion using unified memory\n";
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";     
        // Run the algorithm for several execution configuration sintaxes (threads per block from 32(==1 warp) to 256)
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            // Get number of blocks from the max. number of threads per SM and the number of SMs
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_RGB2YCbCrTime = GPU_ColorConversionUnifiedBenchmark(blocks, threadsPerBlock, inputImgPath);
            printBenchmark(cuda_RGB2YCbCrTime, cpu_test, cpu_RGB2YCbCrTime);
        }
        

        // RGB <-> YCbCr using pageable memory
        cout<<"\n\nColor Conversion using pageable memory\n";
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n"; 
        // Run the algorithm for several execution configuration sintaxes (threads per block from 32(==1 warp) to 256)    
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            // Get number of blocks from the max. number of threads per SM and the number of SMs
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_RGB2YCbCrTime = GPU_ColorConversionPageableBenchmark(blocks, threadsPerBlock, inputImgPath);
            printBenchmark(cuda_RGB2YCbCrTime, cpu_test, cpu_RGB2YCbCrTime);
        }


        // RGB <-> YCbCr using pinned memory
        cout<<"\nColor Conversion using pinned memory\n";
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";     
        // Run the algorithm for several execution configuration sintaxes (threads per block from 32(==1 warp) to 256)
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            // Get number of blocks from the max. number of threads per SM and the number of SMs
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_RGB2YCbCrTime = GPU_ColorConversionPinnedBenchmark(blocks, threadsPerBlock, inputImgPath);
            printBenchmark(cuda_RGB2YCbCrTime, cpu_test, cpu_RGB2YCbCrTime);
        }

        // Histogram normalization and equalization 
        cout<<"\nHistograms on RGB-images\n";
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";   
        // Run the algorithm for several execution configuration sintaxes (threads per block from 32(==1 warp) to 256)
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            // Get number of blocks from the max. number of threads per SM and the number of SMs
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;    
            cuda_RGB_histogramTime = GPU_RGB_HistogramOperationsBenchmark(blocks, threadsPerBlock, inputImgPath);
            printBenchmark(cuda_RGB_histogramTime, cpu_test, cpu_RGB_histogramTime);
        }

    }
    // If the input image is a grey-value-picture
    else if(inputImgExtension == "pgm")
    {
        // Histogram normalization and equalization
        cout<<"\n\nHistograms on grey value images\n";
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";   
        // Run the algorithm for several execution configuration sintaxes (threads per block from 32(==1 warp) to 256)
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            // Get number of blocks from the max. number of threads per SM and the number of SMs
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

        // RGB <-> YCbCr using unified memory
        cout<<"\n\nColor Conversion using Unified memory\n";
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";     
        // Run the algorithm for several execution configuration sintaxes (threads per block from 32(==1 warp) to 256)
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            // Get number of blocks from the max. number of threads per SM and the number of SMs
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_RGB2YCbCrTime = 0;

            // Iterate over the default set of benchmark images: 21 pictures with filenames 1.ppm,2.ppm,3.ppm ... 20.ppm
            for(int i = 0; i < 21; i++ )
            {
                defaultPath = inputImgPath + std::to_string(i) +".ppm";
                cuda_RGB2YCbCrTime += GPU_ColorConversionUnifiedBenchmark(blocks, threadsPerBlock, defaultPath);
            }

            printBenchmark(cuda_RGB2YCbCrTime, cpu_test, cpu_RGB2YCbCrTime);
        }
        
        // RGB <-> YCbCr using pageable memory
        cout<<"\n\nColor Conversion using pageable memory\n";
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";     
        // Run the algorithm for several execution configuration sintaxes (threads per block from 32(==1 warp) to 256)
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            // Get number of blocks from the max. number of threads per SM and the number of SMs
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_RGB2YCbCrTime = 0;


            // Iterate over the default set of benchmark images: 21 pictures with filenames 1.ppm,2.ppm,3.ppm ... 20.ppm
            for(int i = 0; i < 21; i++ )
            {
                defaultPath = inputImgPath + std::to_string(i) +".ppm";
                cuda_RGB2YCbCrTime += GPU_ColorConversionPageableBenchmark(blocks, threadsPerBlock, defaultPath);
            }

            printBenchmark(cuda_RGB2YCbCrTime, cpu_test, cpu_RGB2YCbCrTime);
        }

        // RGB <-> YCbCr using pinned memory
        cout<<"\nColor Conversion using pinned memory\n";
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";    
        // Run the algorithm for several execution configuration sintaxes (threads per block from 32(==1 warp) to 256) 
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            // Get number of blocks from the max. number of threads per SM and the number of SMs
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_RGB2YCbCrTime = 0;


            // Iterate over the default set of benchmark images: 21 pictures with filenames 1.ppm,2.ppm,3.ppm ... 20.ppm
            for(int i = 0; i < 21; i++ )
            {
                defaultPath = inputImgPath + std::to_string(i) +".ppm";
                cuda_RGB2YCbCrTime += GPU_ColorConversionPinnedBenchmark(blocks, threadsPerBlock, defaultPath);
            }
            printBenchmark(cuda_RGB2YCbCrTime, cpu_test, cpu_RGB2YCbCrTime);
        }

        // Histogram normalization and equalization (RGB images)
        cout<<"\nHistograms on RGB-images\n";
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";   
        // Run the algorithm for several execution configuration sintaxes (threads per block from 32(==1 warp) to 256)
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            // Get number of blocks from the max. number of threads per SM and the number of SMs
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock;
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_RGB_histogramTime = 0;    

            // Iterate over the default set of benchmark images: 21 pictures with filenames 1.ppm,2.ppm,3.ppm ... 20.ppm
            for(int i = 0; i < 21; i++ )
            {
                defaultPath = inputImgPath + std::to_string(i) +".ppm";
                cuda_RGB_histogramTime += GPU_RGB_HistogramOperationsBenchmark(blocks, threadsPerBlock, defaultPath);
            }
            printBenchmark(cuda_RGB_histogramTime, cpu_test, cpu_RGB_histogramTime);
        }

        // Histogram normalization and equalization (GVPs)
        cout<<"\nHistograms on grey value images\n";
        cout<<"Blocks \tThreads\t";
        cout<<"GPU[ms]\tCPU[ms]\tSpeedUp(Cuda-Device with respect to CPU)\n";   
        // Run the algorithm for several execution configuration sintaxes (threads per block from 32(==1 warp) to 256)
        for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
        {
            // Get number of blocks from the max. number of threads per SM and the number of SMs
            int blocks = numSM*maxThreadsPerSM/threadsPerBlock; 
            cout<< blocks<<'\t' << threadsPerBlock;
            cuda_GVP_histogramTime = 0;

            // Iterate over the default set of benchmark images: 21 GVPs with filenames 1.pgm,2.pgm,3.pgm ... 20.pgm
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
