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
void ColorConversionBenchmarking(dim3 blocks, dim3 threadsPerBlock)
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
        string path = "../../TestImages/Benchmark/"+ std::to_string(i) +".ppm";
        Image test(path.data());
        double miliseconds = 0;

        // CUDA-Device
        miliseconds += test.dev_rgb2yuv(blocks,threadsPerBlock);
        miliseconds += test.dev_yuv2rgb(blocks,threadsPerBlock);
        cuda_RGB2YCbCrTime += miliseconds;
        //cout << test.getRows() << '\t' << test.getCols() << '\t' << miliseconds << '\t';

        N += 3*test.getRows()*test.getCols();

        // CPU
        start_t = clock();
        test.host_rgb2yuv();
        test.host_yuv2rgb();
        stop_t = clock();
        miliseconds +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
        cpu_RGB2YCbCrTime+= miliseconds;
        
        //cout << '\t' << miliseconds <<'\n';

    }

    cout <<'\n';
    cout <<"CUDA RGB <-> YCrCb Conversion: " << cuda_RGB2YCbCrTime  << " ms\n";
    cout <<"CPU  RGB <-> YCrCb Conversion: " << cpu_RGB2YCbCrTime << " ms\n";
    cout <<"CUDA RGB <-> YCrCb Conversion: " << cuda_RGB2YCbCrTime /N << " ms/pixel\n";
    cout <<"CPU  RGB <-> YCrCb Conversion: " << cpu_RGB2YCbCrTime/N << " ms/pixel\n";
    
    double speedUp = 100*cpu_RGB2YCbCrTime/cuda_RGB2YCbCrTime ;
    cout << "\nSpeedup(Cuda-Device with respect to CPU): " << speedUp << "%\n\n";

    // Theoretical Memory Throughput[Gb/s]: 2*CUDA-Device clock-Rate[Hz] * busWidth[bytes]/1.0e9
    int clockRate_kHz;
	int busWidth_bits;
	cudaDeviceGetAttribute(&clockRate_kHz, cudaDevAttrMemoryClockRate, 0);
	cudaDeviceGetAttribute(&busWidth_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
	double theorMemThroughput= 2.0*clockRate_kHz*(busWidth_bits/8.0)/1.0e6;

    //Effective Memory Throughput[Gb/s]: Total number of Gbytes * 2(read/write) / time[ms]
    double effMemThroughput = (N*2/cuda_RGB2YCbCrTime )/1.0e6;

    cout <<"Theoretical Memory Throughput[GB/s]:\t\t" << theorMemThroughput << '\n';
    cout <<"Effective Memory Throughput[GB/s]:\t\t" << effMemThroughput << '\n';
}

void RGB_HistogramOperationsBenchmarking(dim3 blocks, dim3 threadsPerBlock)
{
    clock_t start_t, stop_t;
    
    long N = 0; 
    
    double cpu_histogramTime = 0;
    double cuda_histogramTime = 0; 
 
    cout << "RGB:\n";

    for (int i = 0; i < 21; i++)
    {   
        cout.precision(5);
        string path = "../../TestImages/Benchmark/"+ std::to_string(i) +".ppm";
        Image test(path.data());
        Histogram hist(test);
        double miliseconds = 0;

        // CUDA-Device
        miliseconds += hist.dev_normalize(blocks,threadsPerBlock);
        miliseconds += hist.dev_equalize(blocks,threadsPerBlock);
        cuda_histogramTime+= miliseconds;

        N += 3*test.getRows()*test.getCols();

        // CPU
        start_t = clock();
        hist.host_normalize();
        hist.host_equalize();
        stop_t = clock();
        miliseconds +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
        cpu_histogramTime+= miliseconds;

    }

    cout <<'\n';
    cout <<"CUDA Histogram Equalisierung: " << cuda_histogramTime << " ms\n";
    cout <<"CPU  Histogram Equalisierung: " << cpu_histogramTime << " ms\n";
    cout <<"CUDA Histogram Equalisierung: " << cuda_histogramTime/N << " ms/pixel\n";
    cout <<"CPU  Histogram Equalisierung: " << cpu_histogramTime/N << " ms/pixel\n";
    
    double speedUp = 100*cpu_histogramTime/cuda_histogramTime;
    cout << "\nSpeedup(Cuda-Device with respect to CPU): " << speedUp << "%\n\n";
}

void GVP_HistogramOperationsBenchmarking(dim3 blocks, dim3 threadsPerBlock)
{
    clock_t start_t, stop_t;
    
    long N = 0; 
    
    double cpu_histogramTime = 0;
    double cuda_histogramTime = 0; 
 
    cout << "Grey Value:\n";

    for (int i = 0; i < 21; i++)
    {   
        cout.precision(5);
        string path = "../../TestImages/Benchmark/"+ std::to_string(i) +".pgm";
        Image test(path.data());
        Histogram hist(test);
        double miliseconds = 0;

        // CUDA-Device
        miliseconds += hist.dev_normalize(blocks,threadsPerBlock);
        miliseconds += hist.dev_equalize(blocks,threadsPerBlock);
        cuda_histogramTime+= miliseconds;

        N += 3*test.getRows()*test.getCols();

        // CPU
        start_t = clock();
        hist.host_normalize();
        hist.host_equalize();
        stop_t = clock();
        miliseconds +=1000.0*((double)stop_t - (double)start_t)/CLOCKS_PER_SEC;
        cpu_histogramTime+= miliseconds;

    }

    cout <<'\n';
    cout <<"CUDA Histogram Equalisierung: " << cuda_histogramTime << " ms\n";
    cout <<"CPU  Histogram Equalisierung: " << cpu_histogramTime << " ms\n";
    cout <<"CUDA Histogram Equalisierung: " << cuda_histogramTime/N << " ms/pixel\n";
    cout <<"CPU  Histogram Equalisierung: " << cpu_histogramTime/N << " ms/pixel\n";
    
    double speedUp = 100*cpu_histogramTime/cuda_histogramTime;
    cout << "\nSpeedup(Cuda-Device with respect to CPU): " << speedUp << "%\n\n";
}

int main()
{
    
    int numSM;
    int maxThreadsPerSM;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

    for(int threadsPerBlock = 32; threadsPerBlock<512; threadsPerBlock*=2)
    {
        int blocks = numSM*maxThreadsPerSM/threadsPerBlock;

        cout<<"\nBlocks :\t" << blocks <<"\tThreads per Block:\t" << threadsPerBlock << "\n";
        ColorConversionBenchmarking(blocks, threadsPerBlock);
        GVP_HistogramOperationsBenchmarking(blocks, threadsPerBlock);     
        RGB_HistogramOperationsBenchmarking(blocks, threadsPerBlock);
    }
 
    return EXIT_SUCCESS;
}
