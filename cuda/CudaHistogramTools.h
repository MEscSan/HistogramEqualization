/*
    Small class to get, normalize and equalize histograms. For image representation it relies on the Image class (see CudaImageTools for implementation details)
    Histograms, Equalization and Normalization can be run either on the cpu or the gpu

    Remark: all methods and functions with the prefix host_ are run on the cpu
            all methods and functions with the prefic dev_ are either run on the CUDA device or contain the memory allocation, copy and kernel call to run 
            algorithms on the CUDA device
*/

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

class Histogram
{
    private:

        /* 
            Properties:
            -> number of grey values used to build the histogram
            -> pointers to the "raw" histogram (number of pixels containing a certain grey value) in both host and CUDA device (dev) memory
            -> pointers to the cumulative histogram (normalized to the interval [0,1]) in both host and CUDA device (dev) memory
            -> pointers to the lookup table with the new values corresponding to each of the original grey values after normalization/equalization in both host and CUDA device (dev) memory
            -> pointer to the pixels of the source image on the CUDA device memory 
            -> minimun and maximum grey value in the source image
            -> source image as (pointer to) Image-object
	    */
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

        // Histogram calculation is run by default whithin the constructor, the input parameter host allows to decide
        // whether to run it on the CUDA device (0, default) or on the cpu
        Histogram(Image& src, int host=0);

        // Get both histogram and cumulative histogram of the source Image
        float dev_getHistogram(dim3 blocks = 48, dim3 threadsPerBlock = 128);
        void host_getHistogram();
        
        // Display histogram (vertically) on a stream (f.e. a file or the console)
        void display(ostream& output = cout);    
        
        // Equalize source Image and histogram 
        float dev_equalize(dim3 blocks = 48, dim3 threadsPerBlock = 128);
        void host_equalize();
        
        // Normalize source Image and histogram
        float dev_normalize(dim3 blocks = 48, dim3 threadsPerBlock = 128);
        void host_normalize();

        // Save histogram, cumulative histogram and vertical graphical representation into a .txt file under path
        void save(string path);

};

// Gets the biggest grey value of an image (int and unsigned char verion)
inline int getMax(int* arrayPtr, int arraySize, colorSpace cs = colorSpace::gvp)
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
    // If source is color Image (YCbCr) consider only the Y-Channel
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

inline unsigned char getMax(unsigned char* arrayPtr, int arraySize, colorSpace cs = colorSpace::gvp)
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
    // If source is color Image (YCbCr) consider only the Y-Channel
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

// Get the smallest grey value of an image
inline unsigned char getMin(unsigned char* arrayPtr, int arraySize, colorSpace cs = colorSpace::gvp)
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
    // If source is color Image (YCbCr) consider only the Y-Channel
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

__global__ void getHistogram(unsigned char* pixelPtr, int* values, double* valuesCumulative, unsigned char* lookUpTable,  int rows, int cols, colorSpace color);

// Implements a one-dimensional version of the local-histograms-kernel proposed in https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
__global__ void partialHistograms(unsigned char* pixelPtr, int* g_partialHistograms,int numValues, int rows, int cols, int channels);

// Implements partial-histograms-reduction kernel as proposed in https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
__global__ void globalHistogram(int* g_partialHistograms, int* histogram, int numValues, int numPartialHistograms);

// Algorithm and code proposed in "GPU Gems 3", chapter 39 (Parallel Prefix Sum (Scan) with CUDA) for parallelization of prefix sum
__global__ void partialCumulativeHistograms(int* values, int* g_partialCumulative, int* sums, int n, int nPartial);
__global__ void auxiliaryCumulativeHistogram(int* sums, int n);
__global__ void globalCumulativeHistogram(int* g_partialCumulative, int* sums, double* dev_valuesCumulative, int numValues, int nPartial, int rows, int cols);

// Creates the lookup-table for histogram normalizaton
__global__ void normalizationLookUpTable(unsigned char* dev_lookUpTable, int numValues, unsigned char max, unsigned char min);
__global__ void  equalizationLookUpTable(unsigned char* dev_lookUpTable, double* dev_valuesCumulative, int numValues);

// Updates Pixel-values using the lookup Table
__global__ void updatePixelsFromLookUp( unsigned char* pixelPtr, unsigned char* dev_lookUpTable, int rows, int cols, int channels);
