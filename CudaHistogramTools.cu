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

Histogram::Histogram(Image& src, int host)
    :_src(src)
{
    _numValues = _src.getNumberOfValues();
    _dev_pixels = _src.getDevPixelPtr();
    _host_values = new int[_numValues];
    _host_valuesCumulative = new double[_numValues];
    _host_lookUpTable = new unsigned char[_numValues]; 

    if(host)
    {
        host_calculate();
    }
    else
    {
        calculate();
    }

}

void Histogram::calculate(dim3 blocks, dim3 threadsPerBlock)
{
    int rows = _src.getRows();
    int cols = _src.getCols();
    int channels = _src.getNumberOfChannels();
    double numPixels = rows*cols*channels;

    unsigned char* pixelPtr = (unsigned char*)_src.getPixelPtr();
    int* g_partialHistograms;
    int* g_partialCumulative;
    int* sums;

    //Reset histogram- and cumulative-histogram-array with 0s
    for (int i = 0; i < _numValues; i++)
    {
        _host_values[i]=0;
        //_host_lookUpTable[i]=0;
        _host_valuesCumulative[i]=0;

    }

    // RGB-Images are transformed to YUV-Color space; the histogram-class only takes the y-channel (luminance) into account
    if(_src.getColorSpace()== colorSpace::rgb)
    {
        _src.rgb2yuv();
    }

    // Allocate device memory:
    gpuErrchk( cudaMalloc((void**)& _dev_pixels, numPixels*sizeof(unsigned char)));
    gpuErrchk( cudaMalloc((void**)& _dev_values, _numValues*sizeof(int)));
    gpuErrchk( cudaMalloc((void**)& _dev_valuesCumulative, _numValues*sizeof(double)));    
    gpuErrchk( cudaMalloc((void**)& g_partialHistograms, _numValues*blocks.x*sizeof(int)));
 
    //gpuErrchk( cudaMalloc((void**)& _dev_lookUpTable, _numValues*sizeof(unsigned char)));

    gpuErrchk( cudaMemcpy(_dev_pixels, _src.getPixelPtr(), numPixels*sizeof(unsigned char), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(_dev_values, _host_values, _numValues*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(_dev_valuesCumulative, _host_valuesCumulative, _numValues*sizeof(double), cudaMemcpyHostToDevice));
    //gpuErrchk( cudaMemcpy(_dev_lookUpTable, _host_lookUpTable, _numValues*sizeof(unsigned char), cudaMemcpyHostToDevice));
   
    // Histogram
    partialHistograms<<<blocks, threadsPerBlock, _numValues*sizeof(int)>>>(_dev_pixels, g_partialHistograms, _numValues, rows, cols, channels);
    gpuErrchk(cudaGetLastError());
    globalHistogram<<<blocks, threadsPerBlock>>>(g_partialHistograms, _dev_values, _numValues, blocks.x);
    gpuErrchk(cudaGetLastError());

    // Cumulative Histogram:
    // In the next Kernel, each block scans numValues/n elements, n is the size of the padded array (in the case that numValues is not a multiple of the number of blocks)
    int n = _numValues;
    // If the number of array-elements is not a multiple of the number of blocks it is padded to the next one
    if(n%blocks.x!=0)
    {
        n += blocks.x - n%blocks.x;
    }

   gpuErrchk( cudaMalloc((void**)& g_partialCumulative, _numValues*blocks.x*sizeof(int)));
    gpuErrchk( cudaMalloc((void**)& sums, n*sizeof(int)/128));

    //partialCumulativeHistograms<<<blocks, threadsPerBlock, n*sizeof(int)/blocks.x>>>(_dev_values, g_partialCumulative, sums, _numValues, n);
    partialCumulativeHistograms<<<n/128, 128/2, 128*sizeof(int)>>>(_dev_values, g_partialCumulative, sums, _numValues, n);
    gpuErrchk( cudaGetLastError());
    globalCumulativeHistogram<<<blocks, threadsPerBlock>>>(g_partialCumulative, sums, _dev_valuesCumulative, _numValues,n, rows, cols);
    gpuErrchk( cudaGetLastError());

    gpuErrchk(cudaMemcpy(_host_values, _dev_values, _numValues*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(_host_valuesCumulative, _dev_valuesCumulative, _numValues*sizeof(double), cudaMemcpyDeviceToHost));
    //gpuErrchk(cudaMemcpy(_host_lookUpTable, _dev_lookUpTable, _numValues*sizeof(unsigned char), cudaMemcpyDeviceToHost));

    cudaFree(_dev_pixels);
    cudaFree(_dev_values);
    cudaFree(_dev_valuesCumulative);
    cudaFree(g_partialCumulative);
    cudaFree(sums);
    cudaFree(g_partialHistograms);
    //cudaFree(_dev_lookUpTable);   
        
    // Cumulative histogram (serially, for the moment)
    /*double cdfval=0;

    numPixels = rows*cols;
        
    for (int i = 0; i < _numValues; i++)
    {
        cdfval += (_host_values[i])/(double)numPixels;
        _host_valuesCumulative[i] = cdfval;
    }*/
}

void Histogram::host_calculate()
{
    int rows = _src.getRows();
    int cols = _src.getCols();
    int channels = _src.getNumberOfChannels();
    double numPixels = rows*cols*channels;

    unsigned char* pixelPtr = (unsigned char*)_src.getPixelPtr();
    unsigned char value = 0;
 
    //Reset _host_values-array and _lookupTable with 0s
    for (int i = 0; i < _numValues; i++)
    {
        _host_values[i]=0;
        //_host_lookUpTable[i]=0;
        _host_valuesCumulative[i]=0;
    }

    // In order to make the algorithm robust against multi-channel images (converted gvp- and color images)
    // only the first image channel is used for the histogram 

    // RGB-Images are transformed to YUV-Color space; the histogram-class only takes the y-channel (luminance) into account
    if(_src.getColorSpace()== colorSpace::rgb)
    {
        _src.host_rgb2yuv();
    }

    // Histogram
    for (int i = 0; i < numPixels; i+=channels)
    {
        value = pixelPtr[i];
        _host_values[value]++; 
    }

    // Cumulative histogram

    // For the cumulative histogram the number of channels is no longer relevant
    numPixels = rows*cols;
    double cdfval=0;

    for (int i = 0; i < _numValues; i++)
    {
        cdfval += (_host_values[i])/(double)numPixels;
        _host_valuesCumulative[i] = cdfval;
    }
}

void Histogram::display(ostream& output)
{
    int maxVal = getMax(_host_values, _numValues, _src.getColorSpace());
    
    int normValue = 0;

    for (int i = 0; i < _numValues; i++)
    {
        output << i << "\t|"; 
        normValue = (int)200*(_host_values[i]/(float)maxVal);
        for (int j = 0; j < normValue; j++)
        {
            output << '*';
        }
        output << '\n';
    }
    
}

// Source: 2010_Szeleski_Computer Vision, algorithm and Applications, 3.1.4 bzw. 2012_Prince_ComputervisionModelsLearningAndInferenz
void Histogram::host_equalize()
{
    int numPixels = _src.getRows()*_src.getCols()*_src.getNumberOfChannels();
    unsigned char* pixelPtr = (unsigned char*)_src.getPixelPtr();

    // The normalized cumulative histogram is used as a lookup-table to calculate the new color values
    for (int i = 0; i < _numValues; i++)
    {
        _host_lookUpTable[i] = clamp( _numValues*_host_valuesCumulative[i]);

    }
    
    // Equalize image
    //****COLOR SPACE DEPENDENT OPERATION****//
    if(_src.getColorSpace()== colorSpace::gvp)
    {
        for (int i = 0; i < numPixels; i++)
        {
            unsigned char oldPixelVal = pixelPtr[i];
            unsigned char newPixelVal = _host_lookUpTable[oldPixelVal];
            pixelPtr[i] = newPixelVal; 
        }
    }
    else
    {

        //if(_src.getColorSpace() == colorSpace::rgb)
        //{    
        _src.rgb2yuv();
        //}

        // In color (yuv) images only the y-channel is equalized
        for (int i = 0; i < numPixels; i+=3)
        {
            unsigned char oldPixelVal = pixelPtr[i];
            unsigned char newPixelVal = _host_lookUpTable[oldPixelVal];
            pixelPtr[i] = newPixelVal; 
        }
    }
    //***************************************//

    //Calculate new Histogram
    calculate();

    //Transform the image back to RGB-Space if necessary
    _src.host_yuv2rgb();

}

// Gets the maxmimum histogram-value
int getMax(int* arrayPtr, int arraySize, colorSpace cs)
{
    int maxVal = 0;

    if(cs == colorSpace::gvp)
    {
        for (int i = 0; i < arraySize; i++)
        {
            if(arrayPtr[i] > maxVal)
            {
                maxVal = arrayPtr[i];
            }
        }
    }
    else
    {
        for (int i = 0; i < arraySize; i+=3)
        {
            if(arrayPtr[i] > maxVal)
            {
                maxVal = arrayPtr[i];
            }
        }
    }
    return maxVal;
}

unsigned char getMax(unsigned char* arrayPtr, int arraySize, colorSpace cs)
{
    unsigned char maxVal = 0;

    if(cs == colorSpace::gvp)
    {
        for (int i = 0; i < arraySize; i++)
        {
            if(arrayPtr[i] > maxVal)
            {
                maxVal = arrayPtr[i];
            }
        }
    }
    else
    {
        for (int i = 0; i < arraySize; i+=3)
        {
            if(arrayPtr[i] > maxVal)
            {
                maxVal = arrayPtr[i];
            }
        }
    }
    
    
    return maxVal;
}

unsigned char getMin(unsigned char* arrayPtr, int arraySize, colorSpace cs)
{
    unsigned char minVal = _SC_UCHAR_MAX;

    if(cs == colorSpace::gvp)
    {
        for (int i = 0; i < arraySize; i++)
        {
            if(arrayPtr[i] < minVal)
            {
                minVal = arrayPtr[i];
            }
        }        
    }
    else
    {
        for (int i = 0; i < arraySize; i+=3)
        {
            if(arrayPtr[i] < minVal)
            {
                minVal = arrayPtr[i];
            }
        }
    }
    
    return minVal;
}

// Normalize Histogram, Source:2012_Nixon_FeaturesExtraction, 3.3.2 Histogram normalization
void Histogram::host_normalize()
{
    //Calculate the normalized color-values into a lookup table assuming a minimum value 0 and a maximum equals the number of values
    int rows = _src.getRows();
    int cols = _src.getCols();
    int channels = _src.getNumberOfChannels();
    int numPixels = rows*cols*channels;
    unsigned char* pixelPtr = (unsigned char*)_src.getPixelPtr(); 

    unsigned char maxPixel = getMax(pixelPtr, numPixels, _src.getColorSpace());
    unsigned char minPixel = getMin(pixelPtr, numPixels, _src.getColorSpace());

    // Create Lookup-table
    for (int i = 0; i < _numValues; i++)
    {
        _host_lookUpTable[i] = _numValues*(i - minPixel)/(double)(maxPixel-minPixel);           
        //_host_lookUpTable[i] = (unsigned char)(_numValues*i/(double)(255));
    }
    
    // Normalize image

    //****COLOR SPACE DEPENDENT OPERATION****//
    if(_src.getColorSpace()== colorSpace::gvp)
    {
 
        for (int i = 0; i < numPixels; i++)
        {
            unsigned char oldPixelVal = pixelPtr[i];
            unsigned char newPixelVal = _host_lookUpTable[oldPixelVal];
            pixelPtr[i] = newPixelVal; 
        }
    }
    else
    {
        //if(_src.getColorSpace() == colorSpace::rgb)
        //{
        _src.rgb2yuv();
        //}

        // In color (yuv) images only the y-channel is equalized
        for (int i = 0; i < numPixels; i+=3)
        {
            unsigned char oldPixelVal = pixelPtr[i];
            unsigned char newPixelVal = _host_lookUpTable[oldPixelVal];
            pixelPtr[i] = newPixelVal; 
        }
    }
    //***************************************//
   

    //Calculate new Histogram
    calculate();

    //Transform the image back to RGB-Space if necessary
    _src.yuv2rgb();

}

void Histogram::save(string path)
{
    //Save the histogram into a txt-file
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

// Cuda implementation of the serial histogram algorithm using shared memory and atomic operations
/*__global__ void dev_calculate(unsigned char* pixelPtr, int* values, double* valuesCumulative, unsigned char* lookUpTable, int rows, int cols, colorSpace color)
{
    double numPixels = rows*cols;
    
    extern __shared__ int s_values[];

    extern __shared__ double s_valuesCumulative[];
    //extern __shared__ unsigned char s_lookUpTable[]

    int numValues = sizeof(s_values[])/sizeof(int);

    //Reset values-array and _lookupTable with 0s
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < numValues; i+= blockDim.x*gridDim.x)
    {
        s_values[i]=0;
        s_lookUpTable[i]=0;
        valuesCumulative[i]=0;
    }

    // Get the histogram, depending on the image type

    //****COLOR SPACE DEPENDENT OPERATION****
    if( color == colorSpace::gvp)
    {
        // Shared histogram
        for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < numPixels; i+= blockDim.x*gridDim.x)
        {
            value = pixelPtr[i];
            atomicAdd(&s_values[value], 1);
 
        }
        _syncThreads();

        double cdfval=0;
        
        // Cumulative histogram (scan)
        //...//
        for (int i = 0; i < numValues; i++)
        {
            atomicAdd();
            cdfval += (values[i])/(double)numPixels;
            _host_valuesCumulative[i] = cdfval;
        }
        _syncThreads();
    }
    else
    {
        // RGB-Images are transformed to YUV-Color space; the histogram-class only takes the y-channel (luminance) into account
        
        _src.rgb2yuv();

        
        for (int i = 0; i < numPixels; i+=3)
        {
            value = pixelPtr[i];
            _host_values[value]++; 
        }

        numPixels = rows*cols;
        
        double cdfval=0;
        
        for (int i = 0; i < _numValues; i++)
         {
            cdfval += (_host_values[i])/(double)numPixels;
            _host_valuesCumulative[i] = cdfval;
        }

    }
    //**************************************

};
*/

// Implement a one-dimensional version of the local-histograms-kernel proposed in https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
__global__ void partialHistograms(unsigned char* pixelPtr, int* g_partialHistograms, int numValues, int rows, int cols, int channels)
{
    // Allocate shared memory for partial histogram (one histogram per block)
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
__global__ void partialCumulativeHistograms(int* values, int* g_partialCumulative, int* sums, int numValues, int n)
{
    extern __shared__ int s_partialCumulative[];

    int localThreadIdx = threadIdx.x;
    int localNumThreads = blockDim.x;
    
    //int globalThreadIdx = threadIdx.x + blockIdx.x*blockDim.x;
    //int globalNumThreads = blockDim.x*gridDim.x;

    int offset = 1;
    
    // Initialize shared memory with 0-values
    //for(int i = localThreadIdx; i < n/gridDim.x; i+=localNumThreads)
    for(int i = localThreadIdx; i < 128; i+=localNumThreads)
    {
        s_partialCumulative[i]=0;
    }
    __syncthreads();
    
    // Copy input histogram into shared memory
    //for(int i = globalThreadIdx; i<numValues/2; i += globalNumThreads)
    //for(int i= localThreadIdx; i < n/(2*gridDim.x); i+=localNumThreads )
    for(int i= localThreadIdx; i < 128/2; i+=localNumThreads )
    {
        s_partialCumulative[i*2] = values[2*i + blockIdx.x*n/gridDim.x];
        s_partialCumulative[i*2 + 1] = values[2*i + blockIdx.x*n/gridDim.x + 1];
    }
    __syncthreads();
    
    // Up-Sweep Phase of the Sum-Scan-Algorithm
    //for (int d = (n/gridDim.x)>>1; d > 0; d >>= 1) 
    for (int d = 128>>1; d > 0; d >>= 1) 
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
    
    // Clear the last element  
    if (localThreadIdx == 0) 
    { 
        sums[blockIdx.x] =  s_partialCumulative[127];
        s_partialCumulative[127] = 0; 
    } 
    
    // Down-Sweep Phase of the Sum-Scan-Algorithm
    //for (int d = 1; d < (n/gridDim.x); d *= 2)
    for (int d = 1; d < 128; d *= 2)
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
    g_partialCumulative += blockIdx.x*numValues;
    //for(int i = localThreadIdx; i < n/gridDim.x; i+=localNumThreads)
    for(int i = localThreadIdx; i < 128; i+=localNumThreads)
    {
        g_partialCumulative[i] = s_partialCumulative[i];
    }
}

__global__ void globalCumulativeHistogram(int* g_partialCumulative, int* sums, double* _dev_valuesCumulative, int numValues, int n, int rows, int cols)
{
   //Apply parallel Scan-Sum ALgorithm to the sums-array containing the sums of the partial cumulative histograms using global memory

    int globalThreadIdx = threadIdx.x+ blockIdx.x*blockDim.x;
    int globalNumThreads = blockDim.x*gridDim.x;
    
    int offset = 1;

    // Up-Sweep Phase of the Sum-Scan-Algorithm
    for (int d = gridDim.x >>1; d > 0; d >>= 1) 
    { 
        if (globalThreadIdx < d)    
        { 
            int a = offset*(2*globalThreadIdx+1)-1;     
            int b = offset*(2*globalThreadIdx+2)-1;  
            sums[b] += sums[a];    
        }    
        offset *= 2; 
    } 

   /// Clear the last element  
    if (globalThreadIdx == 0) 
    { 
        sums[gridDim.x - 1] = 0; 
    } 

    // Down-Sweep Phase of the Sum-Scan-Algorithm
    for (int d = 1; d < gridDim.x; d *= 2)
    {      
        offset >>= 1;      
        if (globalThreadIdx< d)      
        { 
            int a = offset*(2*globalThreadIdx+1)-1;     
            int b = offset*(2*globalThreadIdx+2)-1; 
             
            int t = sums[a]; 
            sums[a] = sums[b]; 
            sums[b] += t;       
        } 
    }

    // Add i-th cumulative sums value to the (i+1)th values of the partial cumulative histogram
    //for( int i = globalThreadIdx; i<numValues; i+=globalNumThreads )
    for( int i = globalThreadIdx; i<numValues; i+=globalNumThreads )
    {
        /*if( i< n/gridDim.x )
        {
            _dev_valuesCumulative[i] = n;//g_partialCumulative[i] /(double)( rows*cols );
        }
        else
        {*/
            //_dev_valuesCumulative[i] = (g_partialCumulative[i] + sums[i*gridDim.x/n])/(double) ( rows*cols );
            
            _dev_valuesCumulative[i] = sums[i/128];
            //_dev_valuesCumulative[i] = i*gridDim.x/n;
        //}
    }
}

/*

#define NUM_BANKS 16 
#define LOG_NUM_BANKS 4 
#define CONFLICT_FREE_OFFSET(n) \ ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

__global__ void prescan(float *g_odata, float *g_idata, int n) 
{ 
    extern __shared__ float temp[]; // allocated on invocation 

    int thid = threadIdx.x; 
    int offset = 1; 

    temp[2*thid] = g_idata[2*thid]; // load input into shared memory 
    temp[2*thid+1] = g_idata[2*thid+1]; 
 	
    for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree 
    { 
        __syncthreads();   
        if (thid < d)    
        { 
            int ai = offset*(2*thid+1)-1;     
            int bi = offset*(2*thid+2)-1;  
            temp[bi] += temp[ai];    
        }    
        offset *= 2; 
    } 
    
    if (thid == 0) 
    { 
        temp[n - 1] = 0; 
    } // clear the last element  
 	
    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan 
    {      
        offset >>= 1;      
        __syncthreads();      
        if (thid < d)      
        { 
            int ai = offset*(2*thid+1)-1;     
            int bi = offset*(2*thid+2)-1; 
             
            float t = temp[ai]; 
            temp[ai] = temp[bi]; 
            temp[bi] += t;       
        } 
    }  
    __syncthreads(); 

    g_odata[2*thid] = temp[2*thid]; // write results to device memory    
    g_odata[2*thid+1] = temp[2*thid+1]; 
 	
}

*/