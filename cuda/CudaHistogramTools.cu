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
#include "CudaHistogramTools.h"
#include "CudaErrorHelper.h"

using namespace std;

/* 
    Remark: all methods and functions with the prefix host_ are run on the cpu
            all methods and functions with the prefic dev_ are either run on the CUDA device or contain the memory allocation, copy and kernel call to run 
            algorithms on the CUDA device

            all variables with the prefix g_ refer to global memory in CUDA device
            all variables with the prefix s_ refer to shared memory in CUDA device
            
            execution time = parallel algorithm execution time on CUDA device + device-to-host copy time + host-to-device copy time
*/            
Histogram::Histogram(Image& src, int host)
    :_src(src)
{
    // By default ist the number-of-values parameter in RGB .ppm images "255"
    // For histogram calculations, this value has to be corrected to 256 => [0;255]
    if(_src.getNumberOfValues() == 255)
    {
        _numValues = 256;
    }
    else
    {
        _numValues = _src.getNumberOfValues();
    }

    _dev_pixels = _src.getDevPixelPtr();
    _host_values = new int[_numValues];
    _host_valuesCumulative = new double[_numValues];
    _host_lookUpTable = new unsigned char[_numValues]; 

    // Build histogram and cumulative histogram on the cpu or the CUDA device according to "host" parameter
    if(host)
    {
        host_getHistogram();
    }
    else
    {
        dev_getHistogram();
    }

}

// Build histogram and cumulative histogram on CUDA device
// Return the execution time
float Histogram::dev_getHistogram(dim3 blocks, dim3 threadsPerBlock)
{
    // Initialize timing variables
    float miliseconds = 0, ms1 = 0, ms2 = 0;

    // Initialize CUDA events for timing
    cudaEvent_t start1, start2, stop1, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);

    // Get Image dimensions from source
    int rows = _src.getRows();
    int cols = _src.getCols();
    int channels = _src.getNumberOfChannels();
    double numPixels = rows*cols*channels;

    // Get pointer to Image pixels from source
    unsigned char* pixelPtr = (unsigned char*)_src.getHostPixelPtr();

    // Histogram is built in two steps according to https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
    // Pointer to global CUDA device memory with partial histograms
    int* g_partialHistograms;

    // Cumulative histogram is built in three steps according to "GPU Gems 3", chapter 39 (Parallel Prefix Sum (Scan) with CUDA), section 39.2.4
    // Pointer to global CUDA device memory with partial cumulative histograms 
    int* g_partialCumulative;
    // Pointer to global CUDA device memory with offsets for distributed cumulative histogram calculation
    int* sums;

    //Reset histogram- and cumulative-histogram-array with 0s
    for (int i = 0; i < _numValues; i++)
    {
        _host_values[i]=0;
        _host_lookUpTable[i]=0;
        _host_valuesCumulative[i]=0;

    }

    // RGB-Images are transformed to YCbCr-Color space; the histogram-class only takes the y-channel (luminance) into account
    if(_src.getColorSpace()== colorSpace::rgb)
    {
        miliseconds += _src.dev_rgb2yuv(blocks, threadsPerBlock);
    }

    // Allocate device memory:
    gpuErrchk( cudaMalloc((void**)& _dev_pixels, numPixels*sizeof(unsigned char)));
    gpuErrchk( cudaMalloc((void**)& _dev_values, _numValues*sizeof(int)));
    gpuErrchk( cudaMalloc((void**)& _dev_valuesCumulative, _numValues*sizeof(double)));    
    gpuErrchk( cudaMalloc((void**)& g_partialHistograms, _numValues*blocks.x*sizeof(int)));

    // Begin benchmark
    cudaEventRecord(start1);

    gpuErrchk( cudaMemcpy(_dev_pixels, _src.getHostPixelPtr(), numPixels*sizeof(unsigned char), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(_dev_values, _host_values, _numValues*sizeof(int), cudaMemcpyHostToDevice));
   
    // Build partial histogram
    partialHistograms<<<blocks, threadsPerBlock, _numValues*sizeof(int)>>>(_dev_pixels, g_partialHistograms, _numValues, rows, cols, channels);
    gpuErrchk(cudaGetLastError());
    //  Build global histogram
    globalHistogram<<<blocks, threadsPerBlock>>>(g_partialHistograms, _dev_values, _numValues, blocks.x);
    gpuErrchk(cudaGetLastError());

    // Stop benchmark
    gpuErrchk(cudaMemcpy(_host_values, _dev_values, _numValues*sizeof(int), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop1);

    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&ms1, start1, stop1);

    // Cumulative Histogram:
    // In the next Kernel, each block scans numValues/n elements, n is the number of grey values in the histogram
    // (in the case that numValues is not a multiple of the number of blocks)
    int n = _numValues;
    int nPartial =  128;

    // Allocate device memory
    gpuErrchk( cudaMalloc((void**)& g_partialCumulative, n*sizeof(int)));
    gpuErrchk( cudaMalloc((void**)& sums, n*sizeof(int)/nPartial));

    // Begin benchmark
    cudaEventRecord(start2);

    gpuErrchk( cudaMemcpy(_dev_valuesCumulative, _host_valuesCumulative, _numValues*sizeof(double), cudaMemcpyHostToDevice));
    // Build partial cumulative histograms 
    partialCumulativeHistograms<<<n/nPartial, nPartial/2, nPartial*sizeof(int)>>>(_dev_values, g_partialCumulative, sums, n, nPartial);
    gpuErrchk( cudaGetLastError());
    // Build auxiliary array with the offsets to be appliead to each of the partial cumulative histograms
    auxiliaryCumulativeHistogram<<<1, n/(2*nPartial), n*sizeof(int)/nPartial>>>(sums, n/nPartial);
    gpuErrchk( cudaGetLastError());
    // join all partial cumulatives histogram to a global cumulative histogram
    globalCumulativeHistogram<<<blocks, threadsPerBlock>>>(g_partialCumulative, sums, _dev_valuesCumulative, _numValues, nPartial, rows, cols);
    gpuErrchk( cudaGetLastError());

    gpuErrchk(cudaMemcpy(_host_valuesCumulative, _dev_valuesCumulative, _numValues*sizeof(double), cudaMemcpyDeviceToHost));
    
    // Stop benchmark
    cudaEventRecord(stop2);

    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&ms2, start2, stop2);

    // Free allocated space on CUDA device
    cudaFree(_dev_pixels);
    cudaFree(_dev_values);
    cudaFree(_dev_valuesCumulative);
    cudaFree(g_partialCumulative);
    cudaFree(sums);
    cudaFree(g_partialHistograms);   
    
    miliseconds += ms1 + ms2;

    //Convert back to RGB if necessary
    miliseconds += _src.dev_yuv2rgb(blocks, threadsPerBlock);

    return miliseconds;
}

// Build histogram and cumulative histogram sequentially on the cpu
void Histogram::host_getHistogram()
{
    // Get Image dimensions from source
    int rows = _src.getRows();
    int cols = _src.getCols();
    int channels = _src.getNumberOfChannels();
    double numPixels = rows*cols*channels;
    
    // Get pointer to Image pixels from source
    unsigned char* pixelPtr = (unsigned char*)_src.getHostPixelPtr();
    unsigned char value = 0;
 
    //Reset _host_values-array and _lookupTable with 0s
    for (int i = 0; i < _numValues; i++)
    {
        _host_values[i]=0;
        _host_lookUpTable[i]=0;
        _host_valuesCumulative[i]=0;
    }

    // In order to make the algorithm robust against multi-channel images (converted gvp- and color images)
    // only the first image channel is used for the histogram 

    // RGB-Images are transformed to YUV-Color space; the histogram-class only takes the y-channel (luminance) into account
    if(_src.getColorSpace()== colorSpace::rgb)
    {
        _src.host_rgb2yuv();
    }

    // Build histogram
    for (int i = 0; i < numPixels; i+=channels)
    {
        value = pixelPtr[i];
        _host_values[value]++; 
    }

    // Build normalized cumulative histogram (values in interval [0;1])
    // For the cumulative histogram the number of channels is no longer relevant
    numPixels = rows*cols;
    double cdfval=0;

    for (int i = 0; i < _numValues; i++)
    {
        cdfval += (_host_values[i])/(double)numPixels;
        _host_valuesCumulative[i] = cdfval;
    }

    // Convert back to RGB if necessary
    _src.host_yuv2rgb();
}

// Display histogram (y-axis -> grey value; x-axes -> normalized frequency of each value)
void Histogram::display(ostream& output)
{
    int maxVal = getMax(_host_values, _numValues, _src.getColorSpace());
    
    int normValue = 0;

    for (int i = 0; i < _numValues; i++)
    {
        output << i << "\t|"; 
        normValue = (int)100*(_host_values[i]/(float)maxVal);
        for (int j = 0; j < normValue; j++)
        {
            output << '*';
        }
        output << '\n';
    }
    
}

// Equalize image on CUDA device
// Return execution time
float Histogram::dev_equalize(dim3 blocks, dim3 threadsPerBlock)
{
    // Initialize timing variables
    float miliseconds = 0, ms1 = 0;

    // Initialize CUDA events for timing
    cudaEvent_t start, stop; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Get Image dimensions from Source
    int rows = _src.getRows();
    int cols = _src.getCols();
    int channels = _src.getNumberOfChannels();
    int numPixels = rows*cols*channels;

    // Get pointer to Image pixels
    unsigned char* host_pixelPtr = (unsigned char*)_src.getHostPixelPtr(); 

    //Convert to YCbCr color space if necessary
    if(_src.getColorSpace()==colorSpace::rgb)
    {
        miliseconds += _src.dev_rgb2yuv(blocks, threadsPerBlock);
    }

    // Allocate memory on CUDA device
    gpuErrchk(cudaMalloc((void**)& _dev_lookUpTable, _numValues*sizeof(unsigned char)));
    gpuErrchk(cudaMalloc((void**)& _dev_pixels, numPixels*sizeof(unsigned char)));
    gpuErrchk(cudaMalloc((void**)& _dev_valuesCumulative, _numValues*sizeof(double)));

    // Beginn benchmark 
    cudaEventRecord(start);

    gpuErrchk(cudaMemcpy(_dev_lookUpTable, _host_lookUpTable, _numValues*sizeof(unsigned char), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_dev_pixels, host_pixelPtr, numPixels*sizeof(unsigned char), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_dev_valuesCumulative, _host_valuesCumulative, _numValues*sizeof(double),cudaMemcpyHostToDevice));

    // The normalized cumulative histogram is used as a lookup-table to get the new color values
    equalizationLookUpTable<<<1, 256, _numValues*sizeof(unsigned char)>>>(_dev_lookUpTable, _dev_valuesCumulative, _numValues);
    gpuErrchk(cudaGetLastError());
    // Update Image pixels replacing their values with the ones in the look up table
    updatePixelsFromLookUp<<<blocks, threadsPerBlock>>>(_dev_pixels, _dev_lookUpTable, rows, cols, channels);
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaMemcpy(_host_lookUpTable, _dev_lookUpTable, _numValues*sizeof(unsigned char), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy( host_pixelPtr, _dev_pixels, numPixels*sizeof(unsigned char), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(_host_valuesCumulative, _dev_valuesCumulative, _numValues*sizeof(double), cudaMemcpyDeviceToHost));    
    
    // Stop benchmark
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms1, start, stop);

    // Free allocated CUDA device memory
    cudaFree(_dev_lookUpTable);
    cudaFree(_dev_pixels);
    cudaFree(_dev_valuesCumulative);

    miliseconds += ms1;
    // Get new histogram and cumulative histogram
    miliseconds += dev_getHistogram(blocks, threadsPerBlock);

    //Transform the image back to RGB-Space if necessary
    miliseconds += _src.dev_yuv2rgb(blocks, threadsPerBlock);

    return miliseconds;
}

// Source: 2010_Szeleski_Computer Vision, algorithm and Applications, 3.1.4 
void Histogram::host_equalize()
{
    //Get Image dimensions from source
    int rows = _src.getRows();
    int cols = _src.getCols();
    int channels = _src.getNumberOfChannels();
    int numPixels = rows*cols*channels;

    //get pointer to Image pixels
    unsigned char* pixelPtr = (unsigned char*)_src.getHostPixelPtr();

    // The normalized cumulative histogram is used as a lookup-table to getHistogram the new color values
    for (int i = 0; i < _numValues; i++)
    {
        _host_lookUpTable[i] = host_clamp( _numValues*_host_valuesCumulative[i]);

    }

    // Convert Image to YCbCr if necessary 
    if(_src.getColorSpace()==colorSpace::rgb)
    {
        _src.host_rgb2yuv();
    }
    
    // Update Image pixels replacing their values with the ones in the look up table
    for (int i = 0; i < numPixels; i+=channels)
    {
        unsigned char oldPixelVal = pixelPtr[i];
        unsigned char newPixelVal = _host_lookUpTable[oldPixelVal];
        pixelPtr[i] = newPixelVal; 
    }

    //Get new histogram and cumulative histogram
    host_getHistogram();

    //Transform the image back to RGB-Space if necessary
    _src.host_yuv2rgb();

}

// Normalize histogram and image on CUDA device and return execution time
float Histogram::dev_normalize(dim3 blocks, dim3 threadsPerBlock)
{
    // Initialize timing variables
    float miliseconds=0, ms1 = 0;

    // Initialize CUDA events for benchmarking
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Get Image dimensions from source
    int rows = _src.getRows();
    int cols = _src.getCols();
    int channels = _src.getNumberOfChannels();
    int numPixels = rows*cols*channels;

    // Get pointer to Image pixels
    unsigned char* host_pixelPtr = (unsigned char*)_src.getHostPixelPtr(); 

    //  Get biggest and smallest image value on image (sequentially)
    unsigned char maxPixel = getMax(host_pixelPtr, numPixels, _src.getColorSpace());
    unsigned char minPixel = getMin(host_pixelPtr, numPixels, _src.getColorSpace());

    // Convert to YCbCr if necessary
    if(_src.getColorSpace()==colorSpace::rgb)
    {
        miliseconds += _src.dev_rgb2yuv(blocks, threadsPerBlock);
    }

    // Allocate memory on CUDA device
    gpuErrchk(cudaMalloc((void**)& _dev_lookUpTable, _numValues*sizeof(unsigned char)));
    gpuErrchk(cudaMalloc((void**)& _dev_pixels, numPixels*sizeof(unsigned char)));


    // Begin benchmark    
    cudaEventRecord(start);
    
    gpuErrchk(cudaMemcpy(_dev_lookUpTable, _host_lookUpTable, _numValues*sizeof(unsigned char), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_dev_pixels, host_pixelPtr, numPixels*sizeof(unsigned char), cudaMemcpyHostToDevice));
    // Store the normalized pixel values into  lookup table
    normalizationLookUpTable<<<1, 256, _numValues*sizeof(unsigned char)>>>(_dev_lookUpTable,_numValues, maxPixel, minPixel);
    gpuErrchk(cudaGetLastError());
    // Replace the pixel values in the source image with those of the lookup table
    updatePixelsFromLookUp<<<blocks, threadsPerBlock>>>(_dev_pixels, _dev_lookUpTable, rows, cols, channels);
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaMemcpy(_host_lookUpTable, _dev_lookUpTable, _numValues*sizeof(unsigned char), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy( host_pixelPtr, _dev_pixels, numPixels*sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Stop benchmark
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms1, start, stop);

    // Free CUDA device memory
    cudaFree(_dev_lookUpTable);
    cudaFree(_dev_pixels); 

    miliseconds += ms1;
    
    // Update histogram and cumulative histogram
    miliseconds += dev_getHistogram(blocks, threadsPerBlock);
    // Convert back to RGB if necessary
    miliseconds +=_src.dev_yuv2rgb(blocks, threadsPerBlock);

    return miliseconds;
}

// Normalize Histogram
void Histogram::host_normalize()
{
    // Get Image dimensions from source
    int rows = _src.getRows();
    int cols = _src.getCols();
    int channels = _src.getNumberOfChannels();
    int numPixels = rows*cols*channels;

    // Get pointer to image pixels from source
    unsigned char* pixelPtr = (unsigned char*)_src.getHostPixelPtr(); 

    //  Get biggest and smallest image value on image 
    unsigned char maxPixel = getMax(pixelPtr, numPixels, _src.getColorSpace());
    unsigned char minPixel = getMin(pixelPtr, numPixels, _src.getColorSpace());

    // Fill Lookup-table with normalized grey values
    for (int i = 0; i < _numValues; i++)
    {
        _host_lookUpTable[i] = host_clamp(_numValues*(i - minPixel)/(double)(maxPixel-minPixel));           
    }
    
    // Convert Image to YCrCb if necessary
    if(_src.getColorSpace()== colorSpace::rgb)
    {
        _src.host_rgb2yuv();
    }

    // Update pixels with the values from the lookup table
    for (int i = 0; i < numPixels; i+= channels)
    {
        unsigned char oldPixelVal = pixelPtr[i];
        unsigned char newPixelVal = _host_lookUpTable[oldPixelVal];
        pixelPtr[i] = newPixelVal; 
    }

    //getHistogram new Histogram
    host_getHistogram();

    //Transform the image back to RGB-Space if necessary
    _src.host_yuv2rgb();

}

// Save histogram and cumulative histogram as a .txt file into the give path
void Histogram::save(string path)
{
    path += ".txt";

    ofstream dstFile(path.data());

    dstFile << "->Histogram Values:\n";

    for(int i = 0; i < _numValues; i++)
    {
        dstFile << _host_values[i] << '\n';
    }

    dstFile << "\n-> Cumulative Histogram\n";

    for (int i = 0; i < _numValues; i++)
    {
        dstFile << _host_valuesCumulative[i] << '\n';
    }

    dstFile << "\n->Histogram representation:\n";

    display(dstFile);

    
    dstFile.close ();
}

// Implement a one-dimensional version of the local-histograms-kernel proposed in https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
__global__ void partialHistograms(unsigned char* pixelPtr, int* g_partialHistograms, int numValues, int rows, int cols, int channels)
{

    extern __shared__ int s_partialHistogram[];

    // Local (block-intern) thread index 
    int localThreadIdx = threadIdx.x;
    int localNumThreads = blockDim.x;

    // Global thread index
    int globalThreadIdx = threadIdx.x + blockIdx.x*blockDim.x;
    int globalNumThreads = blockDim.x*gridDim.x;

    // Initialize shared memory with 0-values
    for(int i = localThreadIdx; i < numValues; i+=localNumThreads)
    {
        s_partialHistogram[i]=0;
    }
    __syncthreads();

    int val;
    // Fill partial histograms with atomic operations in shared memory
    for(int i = globalThreadIdx; i<rows*cols; i += globalNumThreads)
    {
        val = pixelPtr[i*channels];
        atomicAdd(&s_partialHistogram[val], 1);
    }
    __syncthreads();
    
    // Partial histogram from s_partialHistogram in g_partialHistograms
    // The array g_partialHistograms has a size of numBlocks*numValues 
    // Point to the section of global memory corresponding to this block
    g_partialHistograms += blockIdx.x*numValues;
    for(int i = localThreadIdx; i < numValues; i+=localNumThreads)
    {
        g_partialHistograms[i] = s_partialHistogram[i];
    } 
}

// Implement partial-histograms-reduction kernel as proposed in https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
__global__ void globalHistogram(int* g_partialHistograms, int* histogram, int numValues, int numPartialHistograms)
{
    int thread = threadIdx.x + blockIdx.x*blockDim.x;
    int numThreads = blockDim.x*gridDim.x;
    int val = 0;

    // Each thread collects all the histogram-values for a certain bin and stores them into the global histogram
    for(int i = thread; i< numValues; i+=numThreads)
    {
        for(int j = 0; j < numPartialHistograms; j++)
        {
            val += g_partialHistograms[i + j*numValues];
        }

        histogram[i] =  val;
    }
}

// Algorithm proposed in "GPU Gems 3", chapter 39 (Parallel Prefix Sum (Scan) with CUDA) for parallelization of prefix sum
// The Kernel assumes that the array size is a multiple of the number of blocks, this assumption must be checked before the kernel call
__global__ void partialCumulativeHistograms(int* values, int* g_partialCumulative, int* sums, int n, int nPartial)
{
    extern __shared__ int s_partialCumulative[];

    // Local (block-intern) thread index 
    int localThreadIdx = threadIdx.x;
    int localNumThreads = blockDim.x;

    int globalThreadIdx = threadIdx.x + blockIdx.x*blockDim.x;

    int offset = 1;
    
    // Initialize shared memory with 0-values
    for(int i = localThreadIdx; i < nPartial; i+=localNumThreads)
    {
        s_partialCumulative[i]=0;
    }
    __syncthreads();
    
    // Copy input histogram into shared memory
    for(int i= localThreadIdx; i < nPartial>>1; i+=localNumThreads )
    {

        s_partialCumulative[i*2] = values[2*globalThreadIdx];
        s_partialCumulative[i*2 + 1] = values[2*globalThreadIdx + 1];

    }
    __syncthreads();
    
    // Up-Sweep Phase of the Sum-Scan-Algorithm
    for (int d = nPartial>>1; d > 0; d >>= 1) 
    { 
        __syncthreads();   
        if (localThreadIdx < d)    
        { 
            int a = offset*(2*localThreadIdx+1)-1;     
            int b = offset*(2*localThreadIdx+2)-1;  

            s_partialCumulative[b] += s_partialCumulative[a];    
        }    
        offset *= 2; 
    } 
    __syncthreads();  

    // Clear the last element  
    if (localThreadIdx == 0) 
    { 
        sums[blockIdx.x] =  s_partialCumulative[nPartial - 1];
        g_partialCumulative[blockIdx.x*nPartial + nPartial - 1] = s_partialCumulative[nPartial - 1];
        s_partialCumulative[nPartial - 1] = 0; 
    } 
    
    // Down-Sweep Phase of the Sum-Scan-Algorithm
    for (int d = 1; d < nPartial; d *= 2)
    {      
        offset >>= 1;      
        __syncthreads();      
        if (localThreadIdx < d)      
        { 
            int a = offset*(2*localThreadIdx+1)-1;     
            int b = offset*(2*localThreadIdx+2)-1; 
             
            int t = s_partialCumulative[a]; 
            s_partialCumulative[a] = s_partialCumulative[b]; 
            s_partialCumulative[b] += t;       
        } 
    }  
    __syncthreads(); 
    
    // Write the partial cumulative sums to global memory analog to the partialHistograms-Kernel
    g_partialCumulative += blockIdx.x*nPartial;
    //for(int i = localThreadIdx; i < n/gridDim.x; i+=localNumThreads)
    for(int i = localThreadIdx; i < nPartial - 1; i+=localNumThreads)
    {

        g_partialCumulative[i] = s_partialCumulative[i+1];
    }
}

// Build the prefix sum of the sums-array from the partialCumulativeHistograms kernel
__global__ void auxiliaryCumulativeHistogram(int* sums,  int n)
{
   //Apply parallel Scan-Sum ALgorithm to the sums-array containing the sums of the partial cumulative histograms using global memory
    extern __shared__ int s_sums[];
    int localThreadIdx = threadIdx.x;
    int localNumThreads = blockDim.x;
    //int globalThreadIdx = threadIdx.x + blockIdx.x*blockDim.x;
    
    // COpy sums-array to shared memory
    for(int i = localThreadIdx; i< n>>1; i+=localNumThreads)
    {
        s_sums[2*i] = sums[2*i];
        s_sums[2*i + 1] = sums[2*i + 1];
    }
    __syncthreads();

    int offset = 1;

    // Up-Sweep Phase of the Sum-Scan-Algorithm
    for (int d = n >>1; d > 0; d >>= 1) 
    { 
        __syncthreads();
        if (localThreadIdx < d)    
        { 
            int a = offset*(2*localThreadIdx+1)-1;     
            int b = offset*(2*localThreadIdx+2)-1;  
            s_sums[b] += s_sums[a];    
        }    
        offset *= 2; 
    } 

   /// Clear the last element  
    if (localThreadIdx == 0) 
    { 
        s_sums[n-1] = 0; 
    }

    // Down-Sweep Phase of the Sum-Scan-Algorithm
    for (int d = 1; d <n; d *= 2)
    {      
        offset >>= 1;
        __syncthreads();      
        if (localThreadIdx< d)      
        { 
            int a = offset*(2*localThreadIdx+1)-1;     
            int b = offset*(2*localThreadIdx+2)-1; 
             
            int t = s_sums[a]; 
            s_sums[a] = s_sums[b]; 
            s_sums[b] += t;       
        } 
    }
    __syncthreads();

    // Copy to global memory
    for(int i=localThreadIdx; i<n; i+=localNumThreads)
    {
        sums[i] = s_sums[i]; 
    }

}

// Use the elements of the sums-array from auxiliaryCumulativeHistogram as an offset for the partial cumulative histograms
__global__ void globalCumulativeHistogram(int* g_partialCumulative, int* sums, double* _dev_valuesCumulative, int numValues, int nPartial, int rows, int cols)
{
    int globalThreadIdx = threadIdx.x + blockIdx.x*blockDim.x; 
    int globalNumThreads = gridDim.x*blockDim.x;

    for( int i = globalThreadIdx; i<numValues; i+= globalNumThreads )
    {
        _dev_valuesCumulative[i] = (g_partialCumulative[i] + sums[i/nPartial])/(double) ( rows*cols );
    }
}

// Assuming the lookup table can have a maximum of 256 values (according to its data-type) the normalization can be carried out by a single block
// Should the grid size be bigger than 1, all further blocks remain unused, as an exchange the calculation can be implemented using only shared-memory 
__global__ void normalizationLookUpTable(unsigned char* dev_lookUpTable, int numValues, unsigned char max, unsigned char min)
{
    extern __shared__ unsigned char s_lookUpTable[];

    // Local (block-intern) thread index 
    int localThreadIdx = threadIdx.x;
    int localNumThreads = blockDim.x;

    for(int i=localThreadIdx; i<numValues; i+=localNumThreads)
    {
        s_lookUpTable[i] = dev_clamp(numValues*(i - min)/(double)(max-min));  
    }
    __syncthreads();

    for(int i=localThreadIdx; i<numValues; i+=localNumThreads)
    {
        dev_lookUpTable[i] = s_lookUpTable[i];  
    }

}

// Fill the lookup table with the values of the cumulative histogram
__global__ void equalizationLookUpTable(unsigned char* dev_lookUpTable, double* dev_valuesCumulative, int numValues)
{
    extern __shared__ unsigned char s_lookUpTable[];

    // Local (block-intern) thread index 
    int localThreadIdx = threadIdx.x;
    int localNumThreads = blockDim.x;

    for(int i=localThreadIdx; i<numValues; i+=localNumThreads)
    {
        s_lookUpTable[i] = dev_clamp(numValues*dev_valuesCumulative[i]);  
    }
    __syncthreads();

    for(int i=localThreadIdx; i<numValues; i+=localNumThreads)
    {
        dev_lookUpTable[i] = s_lookUpTable[i];  
    }
}

// Replace pixel values with the ones in a lookup table
__global__ void updatePixelsFromLookUp( unsigned char* pixelPtr, unsigned char* dev_lookUpTable, int rows, int cols, int channels)
{
    int globalThreadIdx = threadIdx.x + blockIdx.x*blockDim.x;
    int globalNumThreads = gridDim.x*blockDim.x;

    for(int i = globalThreadIdx; i<rows*cols; i+=globalNumThreads)
    {
        int j = i*channels;
        unsigned char oldPixelVal = pixelPtr[j];
        unsigned char newPixelVal = dev_lookUpTable[oldPixelVal];
        pixelPtr[j] = newPixelVal;
    }

}