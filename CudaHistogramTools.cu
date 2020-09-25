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
        _host_lookUpTable[i]=0;
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
    int nPartial =  128;

    gpuErrchk( cudaMalloc((void**)& g_partialCumulative, n*sizeof(int)));
    gpuErrchk( cudaMalloc((void**)& sums, n*sizeof(int)/nPartial));

    //partialCumulativeHistograms<<<blocks, threadsPerBlock, n*sizeof(int)/blocks.x>>>(_dev_values, g_partialCumulative, sums, _numValues, n);

    partialCumulativeHistograms<<<n/nPartial, nPartial/2, nPartial*sizeof(int)>>>(_dev_values, g_partialCumulative, sums, n, nPartial);
    gpuErrchk( cudaGetLastError());
    auxiliaryCumulativeHistogram<<<1, n/(2*nPartial), n*sizeof(int)/nPartial>>>(sums, n/nPartial);
    gpuErrchk( cudaGetLastError());
    globalCumulativeHistogram<<<blocks, threadsPerBlock>>>(g_partialCumulative, sums, _dev_valuesCumulative, _numValues, nPartial, rows, cols);
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
        //cdfval += (_host_values[i])/(double)numPixels;
        cdfval += (_host_values[i]);
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
    host_calculate();

    //Transform the image back to RGB-Space if necessary
    _src.host_yuv2rgb();

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
__global__ void partialCumulativeHistograms(int* values, int* g_partialCumulative, int* sums, int n, int nPartial)
{
    extern __shared__ int s_partialCumulative[];

    int localThreadIdx = threadIdx.x;
    int globalThreadIdx = threadIdx.x + blockIdx.x*blockDim.x;
    int localNumThreads = blockDim.x;

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

        /*int a = i + CONFLICT_FREE_OFFSET(i);
        int b = i + nPartial>>1;
        b += CONFLICT_FREE_OFFSET(b);
        
        s_partialCumulative[a] = values[globalThreadIdx];
        s_partialCumulative[b] = values[globalThreadIdx + nPartial/2];*/
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

            /*a += CONFLICT_FREE_OFFSET(a);
            b += CONFLICT_FREE_OFFSET(b);*/

            s_partialCumulative[b] += s_partialCumulative[a];    
        }    
        offset *= 2; 
    } 
    
    // Clear the last element  
    if (localThreadIdx == 0) 
    { 
        sums[blockIdx.x] =  s_partialCumulative[nPartial - 1];
        g_partialCumulative[blockIdx.x*nPartial + nPartial - 1] = s_partialCumulative[nPartial - 1];
        s_partialCumulative[nPartial - 1] = 0; 
    } 
    
    // Down-Sweep Phase of the Sum-Scan-Algorithm
    //for (int d = 1; d < (n/gridDim.x); d *= 2)
    for (int d = 1; d < nPartial; d *= 2)
    {      
        offset >>= 1;      
        __syncthreads();      
        if (localThreadIdx < d)      
        { 
            int a = offset*(2*localThreadIdx+1)-1;     
            int b = offset*(2*localThreadIdx+2)-1; 
             
            /*a += CONFLICT_FREE_OFFSET(a);
            b += CONFLICT_FREE_OFFSET(b);
            */
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
        /*int a = i + CONFLICT_FREE_OFFSET(i) + 1;
        int b = i + nPartial>>1;
        int bankOffsetB =  CONFLICT_FREE_OFFSET(b);
        g_partialCumulative[i] = s_partialCumulative[a];
        g_partialCumulative[b] = s_partialCumulative[b+bankOffsetB];*/

        g_partialCumulative[i] = s_partialCumulative[i+1];
    }
}

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

__global__ void globalCumulativeHistogram(int* g_partialCumulative, int* sums, double* _dev_valuesCumulative, int numValues, int nPartial, int rows, int cols)
{
    int globalThreadIdx = threadIdx.x + blockIdx.x*blockDim.x; 
    int globalNumThreads = gridDim.x*blockDim.x;

    for( int i = globalThreadIdx; i<numValues; i+= globalNumThreads )
    {
        _dev_valuesCumulative[i] = (g_partialCumulative[i] + sums[i/nPartial])/(double) ( rows*rows );
        //_dev_valuesCumulative[i] = (g_partialCumulative[i]);
    }
}
