#include <cstdlib>
#include <cstdio>
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include "ImageTools.h"
#include "HistogramTools.h"

using namespace std;

Histogram::Histogram(Image& src)
    :_src(src)
{
    _numValues = _src.getNumberOfValues();
    _values = new int[_numValues];
    _valuesCumulative = new double[_numValues];
    _lookUpTable = new unsigned char[_numValues]; 

    calculate();
}

void Histogram::calculate()
{
    int rows = _src.getRows();
    int cols = _src.getCols();
    int channels = _src.getNumberOfChannels();
    double numPixels = rows*cols*channels;

    unsigned char* pixelPtr = (unsigned char*)_src.getPixelPtr();
    unsigned char value = 0;
 
    //Reset _values-array and _lookupTable with 0s
    for (int i = 0; i < _numValues; i++)
    {
        _values[i]=0;
        _lookUpTable[i]=0;
        _valuesCumulative[i]=0;
    }

    // Get the histogram, depending on the image type

    //****COLOR SPACE DEPENDENT OPERATION****//
    if(_src.getColorSpace()== colorSpace::gvp)
    {
        for (int i = 0; i < numPixels; i++)
        {
            value = *pixelPtr;
            _values[value]++; 
            pixelPtr++;
        }

        double cdfval=0;
        
        for (int i = 0; i < _numValues; i++)
         {
            /*if(i==0)
            {
                _valuesCumulative[i] = (double)_values[i]/(double)numPixels;   
            }
            else
            {

                _valuesCumulative[i] = (_values[i] + numPixels*_valuesCumulative[i-1])/numPixels;
            }*/
            cdfval += (_values[i])/(double)numPixels;
            _valuesCumulative[i] = cdfval;
        }
    }
    else
    {
        // RGB-Images are transformed to YUV-Color space; the histogram-class only takes the y-channel (luminance) into account
        //if(_src.getColorSpace() == colorSpace::rgb)
        //{
            _src.rgb2yuv();
        //}
        
        for (int i = 0; i < numPixels; i+=3)
        {
            value = pixelPtr[i];
            _values[value]++; 
        }

        numPixels = rows*cols;
        
        /*for (int i = 0; i < _numValues; i++)
        {
            /*if(i==0)
            {
                _valuesCumulative[i] = (double)_values[i]/(double)numPixels;   
            }
            else
            {
                // Normalize the cumulative distribution function to 255
                _valuesCumulative[i] = (_values[i] + (rows*cols)*_valuesCumulative[i-1])/numPixels;
            }
        }*/
        double cdfval=0;
        
        for (int i = 0; i < _numValues; i++)
         {
            /*if(i==0)
            {
                _valuesCumulative[i] = (double)_values[i]/(double)numPixels;   
            }
            else
            {

                _valuesCumulative[i] = (_values[i] + numPixels*_valuesCumulative[i-1])/numPixels;
            }*/
            cdfval += (_values[i])/(double)numPixels;
            _valuesCumulative[i] = cdfval;
        }

    }
    //***************************************//

}

void Histogram::display(ostream& output)
{
    int maxVal = getMax(_values, _numValues, _src.getColorSpace());
    
    int normValue = 0;

    for (int i = 0; i < _numValues; i++)
    {
        output << i << "\t|"; 
        normValue = (int)200*(_values[i]/(float)maxVal);
        for (int j = 0; j < normValue; j++)
        {
            output << '*';
        }
        output << '\n';
    }
    
}

// Source: 2010_Szeleski_Computer Vision, algorithm and Applications, 3.1.4 bzw. 2012_Prince_ComputervisionModelsLearningAndInferenz
void Histogram::equalize()
{
    int numPixels = _src.getRows()*_src.getCols()*_src.getNumberOfChannels();
    unsigned char* pixelPtr = (unsigned char*)_src.getPixelPtr();

    // The normalized cumulative histogram is used as a lookup-table to calculate the new color values
    for (int i = 0; i < _numValues; i++)
    {
        _lookUpTable[i] = clamp( _numValues*_valuesCumulative[i]);

    }
    
    // Equalize image
    //****COLOR SPACE DEPENDENT OPERATION****//
    if(_src.getColorSpace()== colorSpace::gvp)
    {
        for (int i = 0; i < numPixels; i++)
        {
            unsigned char oldPixelVal = pixelPtr[i];
            unsigned char newPixelVal = _lookUpTable[oldPixelVal];
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
            unsigned char newPixelVal = _lookUpTable[oldPixelVal];
            pixelPtr[i] = newPixelVal; 
        }
    }
    //***************************************//

    //Calculate new Histogram
    calculate();

    //Transform the image back to RGB-Space if necessary
    _src.yuv2rgb();

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
void Histogram::normalize()
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
        _lookUpTable[i] = _numValues*(i - minPixel)/(double)(maxPixel-minPixel);           
        //_lookUpTable[i] = (unsigned char)(_numValues*i/(double)(255));
    }
    
    // Normalize image

    //****COLOR SPACE DEPENDENT OPERATION****//
    if(_src.getColorSpace()== colorSpace::gvp)
    {
 
        for (int i = 0; i < numPixels; i++)
        {
            unsigned char oldPixelVal = pixelPtr[i];
            unsigned char newPixelVal = _lookUpTable[oldPixelVal];
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
            unsigned char newPixelVal = _lookUpTable[oldPixelVal];
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
        dstFile << _values[i] << '\n';
    }

    dstFile << "\n-> Cumulative Histogram\n";

    for (int i = 0; i < _numValues; i++)
    {
        dstFile << _valuesCumulative[i] << '\n';
    }

    dstFile << "\n->Histogram representation:\n";

    display(dstFile);

    
    dstFile.close ();
}
