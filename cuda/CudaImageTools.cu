#include <cstdlib>
#include <cstdio>
#include <string>
#include <cstring>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "CudaImageTools.h"
#include "CudaErrorHelper.h"


using namespace std;

// Generates an Image-Object out of the given path. Invalid file-type leads to an empty 1x1 binary image
Image::Image(string path)
{
        load(path);
}

// Generates an Image-Object with dimensions rowsxcols, default file-type is pbmBin, default pixel-value is 0
Image::Image(int rows, int cols, colorSpace cs, int numValues, fileType type)
{
        _rows = rows;
        _cols = cols;
        _type = type;

        // Images in pbm-format have by default only one color (binary images with two possible values: 0 and 1)
        if(type == fileType::pbmASCII || type == fileType::pbmBin)
        {
                _numValues = 1;
        }
        else
        {
                _numValues = numValues;
        }

        // Grey value pictures have  
        if(cs == colorSpace::gvp)
        {
                _channels = 1;
        }
        else
        {
                _channels = 3;
        }

        _host_pixels = new  unsigned char[_rows*_cols*_channels];
        
        unsigned char* pixelPtr = _host_pixels;

        // Fill the pixels with random values
        // Initialize Random number generator
        srand(time(NULL));

        for (int i = 0; i < _rows*_cols*_channels; i++)
        {
                *pixelPtr= (char) rand()%numValues;
                pixelPtr++;
        }
        
}

// Get- and set-methods for rows, columns, number of colors, color space and file type
/*void Image::setRows(int rows)
{
        _rows = rows;
}*/

int Image::getRows()
{
        return _rows;
} 

/*void Image::setCols(int cols)
{
        _cols = cols;
}*/

int Image::getCols()
{
        return _cols;
}

/*void Image::setType(fileType type)
{
        _type = type;
}*/

fileType Image::getType()
{
        return _type;
}

/*void Image::setNumberOfValues(int numValues)
{
        if(numValues <1)
        {
                _numValues = 1;
        }
        else if(numValues > 255)
        {
                _numValues = 255;
        }
        else
        {
                _numValues = numValues;
        }
}*/

int Image::getNumberOfValues()
{
        return _numValues;
}

unsigned char* Image::getHostPixelPtr()
{
        return _host_pixels;
}

unsigned char* Image::getDevPixelPtr()
{
        return _dev_pixels;
}

colorSpace Image::getColorSpace()
{
        return _colorSpace;
}

// Returns the image-channel at position c as a gvp-Image object
Image Image::getChannel(int c)
{
        Image channelImg = Image(_rows, _cols, colorSpace::gvp, _numValues, fileType::pgmBin);

        unsigned char* dstPtr = (unsigned char*)channelImg.getHostPixelPtr();

        // c can only have values from 0 to channels-1
        c = host_clamp(c, 0, _channels-1);
        for (int i = c; i < _rows*_cols*_channels; i+=_channels)
        {
                *dstPtr = _host_pixels[i];
                dstPtr++;
        }

        return channelImg;
}

// Inserts a gvp-Image into a channel of the right dimensions
// Does nothing if the input Image-object doesn't have the right dimensions or is a color-picture
void Image::setChannel(Image channel, int c)
{
        if(channel.getCols()==_cols, channel.getRows()==_rows && channel.getColorSpace() == colorSpace::gvp)
        {
                unsigned char* srcPtr = (unsigned char*)channel.getHostPixelPtr();

                // c can only have values from 0 to channels-1
                c = host_clamp(c, 0, _channels-1);
                for (int i = c; i < _rows*_cols*_channels; i+=_channels)
                {
                        _host_pixels[i] = *srcPtr;
                        srcPtr++;
                }       
        }

}

// Returns the number of channels (1 for gvp- and 3 for color-pictures)
int Image::getNumberOfChannels()
{
        return _channels;
}

// If the path of a valid file-type given, overwrites the Image-object with the data from the given file, else does nothing
void Image::load(string path)
{
        //Open the source file
        _src = fopen(path.c_str(), "r");    

        header srcHeader = getHeader(_src);
        _rows = srcHeader.rows;
        _cols = srcHeader.cols;
        _type = srcHeader.type;
        _numValues = srcHeader.numColors;
        
        // For the storage allocation the number of channels has to be taken into account, which depends on the file type
        // ppm-files have 3 channels
        if(_type == fileType::ppmASCII || _type == fileType::ppmBin)
        {
                _host_pixels = new unsigned char[_rows*_cols*3];
                _channels = 3;
                // Color pictures (.ppm) are always rgb-pictures
                _colorSpace = colorSpace::rgb;
        }
        else
        {
                _host_pixels = new unsigned char[_rows*_cols];
                _channels = 1;
                _colorSpace = colorSpace::gvp;
        }

        // Store the pixel values into the Image-object
        imageToArray(_rows, _cols, _channels, _type, _src, _host_pixels);

        // close the source file
        fclose(_src);
}

// Save the image object according to its file type in the given path
// Image object of type "invalidType" are stored as txt-files
void Image::save(string path)
{
        if(_type == fileType::pbmASCII || _type == fileType::pbmBin)
        {
                path += ".pbm";
        }
        else if(_type == fileType::pgmASCII || _type == fileType::pgmBin)
        {
                path += ".pgm";
        }
        else if(_type == fileType::ppmASCII || _type == fileType::ppmBin)
        {
                path += ".ppm";
        }
        else
        {
                path += ".txt";
        }

        ofstream dstImg(path.data());

        // Write the header to the new file
        string header = "P";

        // .pbm files contain no information about the number of colors 
        if(_type ==  fileType::pbmASCII || _type == fileType::pbmBin)
        {      
                dstImg <<"P"<< _type << '\n' << _cols << " " << _rows << '\n';       
        }
        else
        {   
                dstImg <<"P"<< _type << '\n' << _cols << " " << _rows << '\n' << _numValues << '\n';
        }

        // Write the pixel values to the new file
        unsigned char* pixelPtr = _host_pixels;
        
        // ASCII-Files
        if(_type==pbmASCII||_type==pgmASCII||_type==ppmASCII)
        {
                for(int i = 0; i < _rows; i++)
                {
                        for (int j = 0; j < _cols*_channels; j++)
                        {
                                dstImg << (int)(*pixelPtr) << " ";
                                pixelPtr++;
                        }
                        
                       dstImg << '\n';
                }
        }
        // Binary files
        else
        {
                for(int i = 0; i < _cols*_rows*_channels; i++)
                {
                        dstImg << *pixelPtr;
                        pixelPtr++;
                }
        }
        
        
        dstImg.close ();
}

// Converts a color-image to gvp
float Image::dev_color2gvp(dim3 blocks, dim3 threadsPerBlock)
{
        // For benchmarking
        float miliseconds = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Allocate Memory in CUDA-Device
        gpuErrchk( cudaMalloc((void**)&_dev_pixels, _rows*_cols*3*sizeof(unsigned char)));

        // Copy pixel array to Cuda device
        gpuErrchk( cudaMemcpy(_dev_pixels, _host_pixels, _rows*_cols*3*sizeof(unsigned char), cudaMemcpyHostToDevice));

        // Run the color-to-gray conversion kernel
        cudaEventRecord(start);
        color2gvp<<<blocks, threadsPerBlock>>>(_dev_pixels, _colorSpace, _rows, _cols);
        cudaEventRecord(stop);

        // Check if the Kernel produced any errors
        gpuErrchk(cudaGetLastError());        
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&miliseconds, start, stop);

        // Copy pixel array back to host
        gpuErrchk(cudaMemcpy(_host_pixels, _dev_pixels, _rows*_cols*3*sizeof(unsigned char), cudaMemcpyDeviceToHost));

        // Free allocated cuda-device memory
        cudaFree(_dev_pixels); 



        _colorSpace = colorSpace::gvp;
        return miliseconds;
}      

// Source: https://en.wikipedia.org/wiki/YUV#Converting_between_Y%E2%80%B2UV_and_RGB
// 2010_ Szeleski_Bildanalyse -> pg. 88
// Transform according to the equation:
//      YUV = M x RGB + K
// Conversion with correction factors, actually RGB <-> YCbCr transform according to Wikipedia
// Conversion Matrix:(CHANGE with Matrix from Nixon 13.3.6.3)
//          0.299        0.587           0.114
//  M =     -0.168736    -0.331264       0.5
//          0.5          -0.418688       -0.081312
//
// Correction vector K:
//          0
//  K =    128
//         128
//
float Image::dev_rgb2yuv(dim3 blocks, dim3 threadsPerBlock)
{
        float miliseconds = 0;
        if(_colorSpace == colorSpace::rgb)
        {
                // For benchmarking
                
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                
                gpuErrchk( cudaMallocManaged((void**)&_dev_pixels, _rows*_cols*3*sizeof(unsigned char)));

                gpuErrchk( cudaMemcpy(_dev_pixels, _host_pixels, _rows*_cols*3*sizeof(unsigned char), cudaMemcpyHostToDevice));

                cudaEventRecord(start);
                rgb2yuv<<< blocks, threadsPerBlock>>>(_dev_pixels, _rows, _cols);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&miliseconds, start, stop); 

                gpuErrchk(cudaGetLastError());
       
                gpuErrchk(cudaMemcpy(_host_pixels, _dev_pixels, _rows*_cols*3*sizeof(unsigned char), cudaMemcpyDeviceToHost));
         
                cudaFree(_dev_pixels);
          
                gpuErrchk(cudaEventDestroy(start));
                gpuErrchk(cudaEventDestroy(stop));
                _colorSpace = colorSpace::yuv;

        } 
        return miliseconds;
}

float Image::dev_rgb2yuv_pinned(dim3 blocks, dim3 threadsPerBlock)
{
        float miliseconds = 0;
        if(_colorSpace == colorSpace::rgb)
        {

                // host pinned memory
                unsigned char* host_pixels_pinned;
                
                // allocate host pinned memory
                gpuErrchk(cudaHostAlloc((void**)&host_pixels_pinned, _rows*_cols*3*sizeof(unsigned char), cudaHostAllocDefault));
                
                // Copy image from pageable to pinned memory
                memcpy(host_pixels_pinned, _host_pixels, _rows*_cols*3*sizeof(unsigned char));

                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                
                gpuErrchk( cudaMalloc((void**)&_dev_pixels, _rows*_cols*3*sizeof(unsigned char)));

                gpuErrchk( cudaMemcpy(_dev_pixels, host_pixels_pinned, _rows*_cols*3*sizeof(unsigned char), cudaMemcpyHostToDevice));

                cudaEventRecord(start);
                rgb2yuv<<< blocks, threadsPerBlock>>>(_dev_pixels, _rows, _cols);
                cudaEventRecord(stop);                
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&miliseconds, start, stop);  

                gpuErrchk(cudaGetLastError());         
                
                gpuErrchk(cudaMemcpy(host_pixels_pinned, _dev_pixels, _rows*_cols*3*sizeof(unsigned char), cudaMemcpyDeviceToHost));
  
                //Copy back pinned memory to pageable memory
                memcpy(_host_pixels, host_pixels_pinned, _rows*_cols*3*sizeof(unsigned char));

                cudaFree(_dev_pixels);
                cudaFreeHost(host_pixels_pinned);

                gpuErrchk(cudaEventDestroy(start));
                gpuErrchk(cudaEventDestroy(stop));
                
                _colorSpace = colorSpace::yuv;

        } 
        return miliseconds;
}

float Image::dev_rgb2hsv(dim3 blocks, dim3 threadsPerBlock)
{
        float miliseconds = 0;

        if(_colorSpace == colorSpace::rgb)
        {
                
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                gpuErrchk( cudaMalloc((void**)&_dev_pixels, _rows*_cols*3*sizeof(unsigned char)));

                gpuErrchk( cudaMemcpy(_dev_pixels, _host_pixels, _rows*_cols*3*sizeof(unsigned char), cudaMemcpyHostToDevice));

                cudaEventRecord(start);
                rgb2hsv<<<blocks, threadsPerBlock>>>(_dev_pixels, _rows, _cols);
                cudaEventRecord(stop);

                gpuErrchk(cudaGetLastError());

                gpuErrchk(cudaMemcpy(_host_pixels, _dev_pixels, _rows*_cols*3*sizeof(unsigned char), cudaMemcpyDeviceToHost));

                cudaFree(_dev_pixels);
                
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&miliseconds, start, stop); 

                _colorSpace = colorSpace::hsv;
        } 

        return miliseconds;
}

// Analog to RGB->YUV transform, solving the equation for the RGB-vector (values smaller than 1e-5 rounded to 0)
//      RGB = ( M^(-1) ) x (YUV - K)
//          1   ~0              1.401999
//  M^-1 =  1   -0.344136       -0.714136
//          1   1.772           ~0
//
float Image::dev_yuv2rgb(dim3 blocks, dim3 threadsPerBlock)
{       
        float miliseconds = 0;

        if(_colorSpace == colorSpace::yuv)
        {
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                gpuErrchk( cudaMalloc((void**)&_dev_pixels, _rows*_cols*3*sizeof(unsigned char)));

                gpuErrchk( cudaMemcpy(_dev_pixels, _host_pixels, _rows*_cols*3*sizeof(unsigned char), cudaMemcpyHostToDevice));

                cudaEventRecord(start);
                yuv2rgb<<<blocks, threadsPerBlock>>>(_dev_pixels, _rows, _cols);
                cudaEventRecord(stop);

                gpuErrchk(cudaGetLastError());

                gpuErrchk(cudaMemcpy(_host_pixels, _dev_pixels, _rows*_cols*3*sizeof(unsigned char), cudaMemcpyDeviceToHost));

                cudaFree(_dev_pixels);
                
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&miliseconds, start, stop); 

                _colorSpace = colorSpace::rgb;
        }

        return miliseconds;
}

float Image::dev_yuv2rgb_pinned(dim3 blocks, dim3 threadsPerBlock)
{       
        float miliseconds = 0;

        if(_colorSpace == colorSpace::yuv)
        {
                // host pinned memory
                unsigned char* host_pixels_pinned;
                
                // allocate host pinned memory
                gpuErrchk(cudaHostAlloc((void**)&host_pixels_pinned, _rows*_cols*3*sizeof(unsigned char), cudaHostAllocDefault));
                
                // Copy image from pageable to pinned memory
                memcpy(host_pixels_pinned, _host_pixels, _rows*_cols*3*sizeof(unsigned char));

                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                gpuErrchk( cudaMalloc((void**)&_dev_pixels, _rows*_cols*3*sizeof(unsigned char)));

                gpuErrchk( cudaMemcpy(_dev_pixels, host_pixels_pinned, _rows*_cols*3*sizeof(unsigned char), cudaMemcpyHostToDevice));

                cudaEventRecord(start);
                yuv2rgb<<<blocks, threadsPerBlock>>>(_dev_pixels, _rows, _cols);
                cudaEventRecord(stop);      
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&miliseconds, start, stop); 

                gpuErrchk(cudaGetLastError());

                gpuErrchk(cudaMemcpy(host_pixels_pinned, _dev_pixels, _rows*_cols*3*sizeof(unsigned char), cudaMemcpyDeviceToHost));

                memcpy(_host_pixels, host_pixels_pinned, _rows*_cols*3*sizeof(unsigned char));

                cudaFree(_dev_pixels);
                cudaFreeHost(host_pixels_pinned);
                
                gpuErrchk(cudaEventDestroy(start));
                gpuErrchk(cudaEventDestroy(stop));

                _colorSpace = colorSpace::rgb;
        }

        return miliseconds;
}

// Converts a color-image to gvp
void Image::host_color2gvp()
{
        // Only make any changes if the image is either rgb or yuv
        // For rgb-Images the average ove all three channels is calculated
        // For yuv-images, the gvp image is the y-channel (the first one)
        if(_colorSpace == colorSpace::rgb)
        {
         
                int numPixels = _rows*_cols*3;
                unsigned char r = 0, g = 0, b = 0, gv =0;

                for (int i = 0; i < numPixels; i+=3)
                {
                        r = _host_pixels[i];
                        g = _host_pixels[i+1];
                        b = _host_pixels[i+2];

                        // Source: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
                        gv = (unsigned char)(0.299*r + 0.587*g + 0.114*b);

                        _host_pixels[i]= gv;
                        _host_pixels[i+1] = gv;
                        _host_pixels[i+ 2] = gv;
                }
                        
        }
        else  if(_colorSpace == colorSpace::yuv)
        {
                int numPixels = _rows*_cols*3;
                unsigned char gv =0;

                for (int i = 0; i < numPixels; i+=3)
                {
                        gv = _host_pixels[i];
                        _host_pixels[i+1] = gv;
                        _host_pixels[i+ 2] = gv;
                }
        }
}

void Image::host_rgb2yuv()
{
        if(_colorSpace == colorSpace::rgb)
        {
                int numPixels = _rows*_cols*3;
                unsigned char r = 0, g = 0, b = 0, y =0, u = 0, v =0;

                for (int i = 0; i < numPixels; i+=3)
                {
                        r = _host_pixels[i];
                        g = _host_pixels[i+1];
                        b = _host_pixels[i+2];

                        // Make sure all values are in the range [0,255]
                        y = host_clamp( 0.299*r + 0.587*g + 0.114*b );
                        u = host_clamp(-0.168736*r - 0.331264*g + 0.500*b + 128);
                        v = host_clamp( 0.500*r - 0.418688*g - 0.081312*b + 128);
                        
                        _host_pixels[i]= y;
                        _host_pixels[i+1] = u;
                        _host_pixels[i+ 2] = v;
                }
    
                _colorSpace = colorSpace::yuv;
        }
}

void Image::host_yuv2rgb()
{       
        if(_colorSpace == colorSpace::yuv)
        {
                int numPixels = _rows*_cols*3;
                unsigned char r = 0, g = 0, b = 0, y =0, u = 0, v =0;

                for (int i = 0; i < numPixels; i+=3)
                {
                        y = _host_pixels[i];
                        u = _host_pixels[i+1];
                        v = _host_pixels[i+2];

                        // Make sure all values are in the range [0,255]
                        r = host_clamp(1*y + 1.401999*(v -128));
                        g = host_clamp(1*y  - 0.344136*(u -128) - 0.714136*(v -128 ));
                        b = host_clamp(1*y + 1.772*(u -128));

                        _host_pixels[i]= r;
                        _host_pixels[i+1] = g;
                        _host_pixels[i+ 2] = b;
                }

                _colorSpace = colorSpace::rgb;
        }
}

__global__ void color2gvp(unsigned char* pixels_ptr, colorSpace color, int rows, int cols)
{ 
        // Only make any changes if the image is either rgb or yuv
        // For rgb-Images the average ove all three channels is calculated
        // For yuv-images, the gvp image is the y-channel (the first one)
        if(color == colorSpace::rgb)
        {
                int numPixels = rows*cols;
                unsigned char r = 0, g = 0, b = 0, gv =0;

                for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < numPixels; i+= blockDim.x*gridDim.x)
                {
                        // Variable j in order to access all three channels of a pixel i
                        int j = i*3;
                        r = pixels_ptr[j];
                        g = pixels_ptr[j+1];
                        b = pixels_ptr[j+2];

                        // Source: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
                        gv = (unsigned char)(0.299*r + 0.587*g + 0.114*b);

                        pixels_ptr[j]= gv;
                        pixels_ptr[j + 1] = gv;
                        pixels_ptr[j + 2] = gv;
                }
        }
        else  if(color == colorSpace::yuv)
        {
                int numPixels = rows*cols;
                unsigned char gv =0;

               for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < numPixels; i+= blockDim.x*gridDim.x)
                {
                        int j = i*3;
                        gv = pixels_ptr[j];
                        pixels_ptr[j + 1] = gv;
                        pixels_ptr[j + 2] = gv;
                }
        }
} 

__global__ void rgb2yuv(unsigned char* pixels_ptr, int rows, int cols)
{

        int numPixels = rows*cols;
        unsigned char r= 0, g = 0, b = 0;

        for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < numPixels; i+= blockDim.x*gridDim.x)
        {
                int j = 3*i;
                r = pixels_ptr[j];
                g = pixels_ptr[j+1];
                b = pixels_ptr[j+2]; 
                
                pixels_ptr[j] = dev_clamp( 0.299*r + 0.587*g + 0.114*b);
                pixels_ptr[j+1] = dev_clamp(-0.168736*r - 0.331264*g + 0.500*b+128);
                pixels_ptr[j+ 2]  = dev_clamp( 0.5*r - 0.419*g - 0.081*b +128);       

        }
}

__global__ void rgb2hsv(unsigned char* pixels_ptr, int rows, int cols)
{

        int numPixels = rows*cols;
        unsigned char r = 0, g = 0, b = 0, h =0, s = 0, v =0, max=0, min=255;

        for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < numPixels; i+= blockDim.x*gridDim.x)
        {

                int j = 3*i;
                r = pixels_ptr[j];
                g = pixels_ptr[j+1];
                b = pixels_ptr[j+2];

                for(int k = 0; k<3; k++)
                {
                        unsigned char val =pixels_ptr[j+k];
                        if(val<min)
                        {
                                min=val;
                        }
                        if(val>max)
                        {
                                max=val;
                        }
                }

                v = max;
                if(v!=0)
                {
                        s = dev_clamp(255*(v - min)/(double)v);
                }

                if(v == r)
                {
                        h=dev_clamp(30*(g-b)/(double)(v - min));
                }
                else 
                if(v == g)
                {
                        h=dev_clamp(60 + 30*(b-r)/(double)(v - min));
                }
                else
                {
                        h = dev_clamp(120 + 30*(r-g)/(double)(v - min));
                }
     

                pixels_ptr[j]= h;
                pixels_ptr[j+1] = s;
                pixels_ptr[j+ 2] = v;
        }
}

__global__ void yuv2rgb(unsigned char* pixels_ptr, int rows, int cols)
{
        int numPixels = rows*cols;
        unsigned char r = 0, g = 0, b = 0, y =0, u = 0, v =0;

        for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < numPixels; i+=blockDim.x*gridDim.x)
        {
                int j = i*3;
                y = pixels_ptr[j];
                u = pixels_ptr[j+1];
                v = pixels_ptr[j+2];

                // Make sure all values are in the range [0,255]
                r = dev_clamp(y + 1.401999*(v-128));
                g = dev_clamp(y  - 0.344136*(u-128) - 0.714136*(v-128));
                b = dev_clamp(y + 1.772*(u-128));
                pixels_ptr[j]= r;
                pixels_ptr[j+1] = g;
                pixels_ptr[j+ 2] = b;
        }
}


