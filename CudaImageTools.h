/*
    Small library with a minimal class and methods required for manipulating, loading and saving images in netpbm formats (.pbm, .pgm, .ppm)
    In this Cuda-version all interactions with external files are run by the host, 
    color space conversions, on the contrary, are run on the Cuda-device
*/
#pragma once 

#include <cstdlib>
#include <cstdio>
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <time.h>


using namespace std;
                
/*

Accepted image formats:
    pbmASCII: binary image, ASCII-stored portable bitmap (pbm)
	pbmBin: binary image, binary-stored portable bitmap (pbm)
	pgmASCII: grey value picture, ASCII-stored portable greymap (pgm)
 	pgmBin: grey value picture, binary-stored portable greymap (pgm)
 	ppmASCII: rgb-image, ASCII-stored portable pixmap (ppm)
 	ppmBin: rgb-image, binary-stored portable pixmap (ppm)

*/                
enum fileType
{    
    invalidType,
    pbmASCII,
    pgmASCII,
    ppmASCII,
    pbmBin, 
    pgmBin,
    ppmBin
}; 

/*
Avaliable color spaces:
    gvp: grey value picture (not an actual color space)
    rgb: three channel images in rgb color space
    yuv: color space for analog television (actually yCrCb), the y-channel contains the picture as gvp and the remaining channels contain the chromatic information
*/
enum colorSpace
{
    gvp,
    rgb,
    yuv,
    hsv
};

/*
Image-header the basic image information
*/
struct header
{
    int rows;
    int cols;
    fileType type;
    int numColors;
};

// Image class to manipulate, load and store images in netpbm formats (.pbm, .pgm, .ppm)
// Assumption: pixels have values between 0 and 255
class Image
{
    private:

        /* 
        Properties:
	    ->rows, columns (cols)
	    ->file type  
	    -> host_pixels: pointer to a one-dimensional byte-array on the host containing the pixel-values of the image as stored in the file
        -> dev_pixels: pointer to a one-dimensional byte-array on the cuda device containing the pixel-values of the image as stored in the file
	    -> number of Values: number of values a single pixel can take (should be between 1 and 255)
        -> file: image file from where to extract the information
        -> color space
	    */

        int _rows;
        int _cols;
        int _channels;
        fileType _type;
        unsigned char* _host_pixels;
        unsigned char* _dev_pixels;
        int _numValues;
        FILE * _src;
        colorSpace _colorSpace;
   
    public: 
        
    //Constructors:

        // Generates an Image-Object with dimensions 1x1, default file-type is pbmBin, default pixel-value is 0
        Image();

		// Generates an Image-Object out of the given path. Invalid file-type leads to an empty 1x1 binary image
        Image(string path);
        
        // Generates an Image-Object with dimensions rowsxcols, default file-type is pbmBin, default pixel-value is 0
	    Image(int rows, int cols, colorSpace cs, int numColors, fileType type = fileType::invalidType);

        
    //Public Methods: 
	        
        // Get- and set-methods for rows, columns, number of colors, color space and file type
        void setRows(int rows);
        int getRows(); 

        void setCols(int cols);
        int getCols();

        
        void setType(fileType type);
        fileType getType();

        void setNumberOfValues(int numValues);
        int getNumberOfValues();

        colorSpace getColorSpace();

        // Returns a pointer to the pixel at (0,0)
        unsigned char* getPixelPtr();
        unsigned char* getDevPixelPtr();
        
        // Returns the image-channel at position c as a gvp-Image object
        Image getChannel(int c);

        // Inserts a gvp-Image into a channel of the right dimensions
        // Does nothing if the input Image-object doesn't have the right dimensions or is a color-picture
        void setChannel(Image channel,int c);

        // Returns the number of channels (1 for gvp- and 3 for color-pictures)
        int getNumberOfChannels();

        // If the path of a valid file-type given, overwrites the Image-object with the data from the given file, else does nothing
        void load(string path);

        // Saves the current image-object as a file under the given path           
        void save(string path); 

        // Colorspace transformations rgb->gvp and rgb <-> yuv transformation
        // The standard methods allocate memory and call the corresponding kernel automatically
        void color2gvp(dim3 blocks = 48, dim3 threadsPerBlock = 128 );
        
        //void gvp2color(dim3 blocks = 128, dim3 threadsPerBlock = 128 );
       
        void rgb2yuv(dim3 blocks = 48, dim3 threadsPerBlock = 128 );
        
        void yuv2rgb(dim3 blocks = 48, dim3 threadsPerBlock = 128 );

        void rgb2hsv(dim3 blocks = 48, dim3 threadsPerBlock = 128 );

        // The "host_" methods run the full color-conversion operation on the CPU
        void host_color2gvp();
        //void host_gvp2color();
        void host_rgb2yuv();
        void host_yuv2rgb();        
      
};

// Functions used inside the Image-methods

// Parse a string in a file to an integer till a whitespace (ASCII Code 32), end of line ('\n') or a comment is found
// Comments a strings beginning with '#' and ending with a '\n' character
inline int parseNumber(FILE * file)
{
    char current_char = fgetc(file);

    // The "-1" value makes sure that at least one non-whitespace character is parsed
    int number = -1;

    // Parse characaters till a whitespace or and end-of-line are reached
    do
    {
        // Comments begin with a '#' (ASCII Code 35); the rest of the line should be ignored
        if(current_char == '#')
        {
           current_char = fgetc(file) ;
           while(current_char !='\n')
           {
                current_char = fgetc(file);
           }  
        }

        // The first character might be a whitespace; in this case it should be ignored
        else if(current_char != 32 && current_char != '\n')
        {
            if(number == -1)
            {
                number = 0;
            }
            number *= 10;
            // "Padding" ASCII-Decimal code to figures (0-9)
            number += (current_char - 48);
        }

        current_char = fgetc(file) ;

    }while((current_char != 32 && current_char != '\n') || number == -1);
    return number;
}

// Returns the header-struct of the image source file
inline header getHeader(FILE* src)
{
    header srcHeader;

    //First read the image format
    string srcType;

    fgets(&srcType[0], 3, src);

    if(memcmp("P1", srcType.data(), 2)==0)
    {
            srcHeader.type = fileType::pbmASCII; 
            srcHeader.cols = parseNumber(src);
            srcHeader.rows = parseNumber(src);
            srcHeader.numColors = 2;
    }
    else if(memcmp("P4", srcType.data(), 2)==0)
    {
            srcHeader.type = fileType::pbmBin; 
            srcHeader.cols = parseNumber(src);
            srcHeader.rows = parseNumber(src);
            srcHeader.numColors = 2;
    }
    else if(memcmp("P2", srcType.data(), 2)==0)
    {
            srcHeader.type = fileType::pgmASCII;
            srcHeader.cols = parseNumber(src);
            srcHeader.rows = parseNumber(src);
            srcHeader.numColors = parseNumber(src);
    }
    else if(memcmp("P5", srcType.data(), 2)==0)
    {
            srcHeader.type = fileType::pgmBin;
            srcHeader.cols = parseNumber(src);
            srcHeader.rows = parseNumber(src);
            srcHeader.numColors = parseNumber(src);
    }        
    else if (memcmp("P3", srcType.data(), 2)==0)
    {
            srcHeader.type = fileType::ppmASCII;
            srcHeader.cols = parseNumber(src);
            srcHeader.rows = parseNumber(src);
            srcHeader.numColors = parseNumber(src);
    }        
    else if(memcmp("P6", srcType.data(), 2)==0)
    {
            srcHeader.type = fileType::ppmBin;
            srcHeader.cols = parseNumber(src);
            srcHeader.rows = parseNumber(src);
            srcHeader.numColors = parseNumber(src);    
    }
    else
    {
            srcHeader.type = fileType::invalidType;
            srcHeader.cols = 1;
            srcHeader.rows = 1;
            srcHeader.numColors = 1;
    }

    return srcHeader;
}

// Reads the pixel values from the original file and stores them in the image-object as a byte-array 
// WARNING: The method assumes that the file pointer points to the first pixel, getHeader must be called before
/*void imageToArray(int rows, int cols, int channels, fileType type, FILE * src, unsigned char* dst);
/__host__ double clamp(double x, double min = 0, double max = 255);
__host__ int clamp (int x, int min = 0, int max = 255);
*/
// FUnctions to be run on the cuda-device
__global__ void dev_color2gvp(unsigned char* pixels_ptr, colorSpace color, int rows, int cols);
__global__ void dev_rgb2yuv(unsigned char* pixels_ptr, int rows, int cols);
__global__ void dev_yuv2rgb(unsigned char* pixels_ptr, int rows, int cols);
__global__ void dev_rgb2hsv(unsigned char* pixels_ptr, int rows, int cols);

// Keeps the given value x within the boundaries [min, max](used mainly in the double-to-byte conversion)  
inline __host__ double clamp(double x, double min=0, double max=255)
{
    double y = x;
    if(x<min)
    {
            y = min;
    }
    else if(x>=max)
    {
            //In order to avoid artifacts because of double to char conversion
            y = max - 0.1;
    }
    return y;
}

inline __host__ int clamp(int x, int min=0, int max=255)
{
    int y = x;
    if(x<min)
    {
            y = min;
    }
    else if(x>=max)
    {
            
            y = max;
    }
    return y;
}

// Makes sure x remains within the interval [min, max]
inline __device__ double dev_clamp(double x, double min = 0, double max = 255)
{

    double y = x;
    if(x<min)
    {
            y = min;
    }
    else if(x>=max)
    {
            //In order to avoid artifacts because of double to char conversion
            y = max - 0.1;
    }
    return y;
}

inline __device__ int dev_clamp (int x, int min = 0, int max = 255)
{
           int y = x;
       if(x<min)
        {
                y = min;
        }
        else if(x>=max)
        {
                
                y = max;
        }
        return y;
}

// Reads the pixel values from the original file and stores them in the image-object as a byte-array 
// WARNING: The method assumes that the file pointer points to the first pixel, getHeader must be called before
inline void imageToArray (int rows, int cols, int channels, fileType type, FILE * src, unsigned char* dst)
{
     char current_char;
    
    if(type==pbmASCII||type==pgmASCII||type==ppmASCII)
    {
        for(int i = 0; i < rows*cols*channels; i++)
        { 
                //Values are parsed like in the header, values over 255 or under 0 are not allowed for pixels
                int character = clamp(parseNumber(src));
                *dst = character;
                dst++;
        }    
    }
    else
    {
        for(int i = 0; i < rows*cols*channels; i++)
        {
                current_char = fgetc(src);
                *dst = current_char;
                dst++;      
        } 
    }
    
}

/*// Implements the Color to Grey-Value conversion in CUDA-Device
inline __global__ void dev_color2gvp(unsigned char* pixels_ptr, colorSpace color, int rows, int cols)
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

// Implements the RGB -> YCbCr conversion in CUDA-Device
inline __global__ void dev_rgb2yuv(unsigned char* pixels_ptr, int rows, int cols)
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
            pixels_ptr[j+ 2] = dev_clamp( 0.5*r - 0.419*g - 0.081*b +128);       

    }
}

// Implements the RGB -> HSV conversion in CUDA-Device
inline __global__ void dev_rgb2hsv(unsigned char* pixels_ptr, int rows, int cols)
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

// Implements the YCbCr -> RGB conversion in CUDA-Device
inline __global__ void dev_yuv2rgb(unsigned char* pixels_ptr, int rows, int cols)
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
}*/