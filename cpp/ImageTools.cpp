#include <cstdlib>
#include <cstdio>
#include <string>
#include <cstring>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "ImageTools.h"

using namespace std;

// Generates an Image-Object with dimensions 1x1, default file-type is pbmBin, default pixel-value is 0
Image::Image()
{
        _rows = 1;
        _cols = 1;
        _type = fileType::pbmASCII;
        _channels =1;
        _pixels = new  unsigned char[_rows*_cols];
        _numValues = 1;
}

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

        _pixels = new  unsigned char[_rows*_cols*_channels];
        
        unsigned char* pixelPtr = _pixels;

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
void Image::setRows(int rows)
{
        _rows = rows;
}

int Image::getRows()
{
        return _rows;
} 

void Image::setCols(int cols)
{
        _cols = cols;
}

int Image::getCols()
{
        return _cols;
}

void Image::setType(fileType type)
{
        _type = type;
}

fileType Image::getType()
{
        return _type;
}

void Image::setNumberOfValues(int numValues)
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
}

int Image::getNumberOfValues()
{
        return _numValues;
}

unsigned char* Image::getPixelPtr()
{
        return _pixels;
}

colorSpace Image::getColorSpace()
{
        return _colorSpace;
}

// Returns the image-channel at position c as a gvp-Image object
Image Image::getChannel(int c)
{
        Image channelImg = Image(_rows, _cols, colorSpace::gvp, _numValues, fileType::pgmBin);

        unsigned char* dstPtr = (unsigned char*)channelImg.getPixelPtr();

        // c can only have values from 0 to channels-1
        c = clamp(c, 0, _channels-1);
        for (int i = c; i < _rows*_cols*_channels; i+=_channels)
        {
                *dstPtr = _pixels[i];
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
                unsigned char* srcPtr = (unsigned char*)channel.getPixelPtr();

                // c can only have values from 0 to channels-1
                c = clamp(c, 0, _channels-1);
                for (int i = c; i < _rows*_cols*_channels; i+=_channels)
                {
                        _pixels[i] = *srcPtr;
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
                _pixels = new unsigned char[_rows*_cols*3];
                _channels = 3;
                // Color pictures (.ppm) are always rgb-pictures
                _colorSpace = colorSpace::rgb;
        }
        else
        {
                _pixels = new unsigned char[_rows*_cols];
                _channels = 1;
                _colorSpace = colorSpace::gvp;
        }

        // Store the pixel values into the Image-object
        imageToArray(_rows, _cols, _channels, _type, _src, _pixels);

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
        unsigned char* pixelPtr = _pixels;
        
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
void Image::color2gvp()
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
                        r = _pixels[i];
                        g = _pixels[i+1];
                        b = _pixels[i+2];

                        // Source: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
                        gv = (unsigned char)(0.299*r + 0.587*g + 0.114*b);

                        _pixels[i]= gv;
                        _pixels[i+1] = gv;
                        _pixels[i+ 2] = gv;
                }
                        
        }
        else  if(_colorSpace == colorSpace::yuv)
        {
                int numPixels = _rows*_cols*3;
                unsigned char r = 0, g = 0, b = 0, gv =0;

                for (int i = 0; i < numPixels; i+=3)
                {
                        gv = _pixels[i];
                        _pixels[i+1] = gv;
                        _pixels[i+ 2] = gv;
                }
        }
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
void Image::rgb2yuv()
{
        if(_colorSpace == colorSpace::rgb)
        {
                int numPixels = _rows*_cols*3;
                unsigned char r = 0, g = 0, b = 0, y =0, u = 0, v =0;

                for (int i = 0; i < numPixels; i+=3)
                {
                        r = _pixels[i];
                        g = _pixels[i+1];
                        b = _pixels[i+2];

                        // Make sure all values are in the range [0,255]
                        y = (unsigned char)clamp( 0.299*r + 0.587*g + 0.114*b );
                        u = (unsigned char)clamp(-0.168736*r - 0.331264*g + 0.500*b + 128);
                        v = (unsigned char)clamp( 0.500*r - 0.418688*g - 0.081312*b + 128);
                        
                        _pixels[i]= y;
                        _pixels[i+1] = u;
                        _pixels[i+ 2] = v;
                }
    
                _colorSpace = colorSpace::yuv;
        }
}


// Analog to RGB->YUV transform, solving the equation for the RGB-vector (values smaller than 1e-5 rounded to 0)
//      RGB = ( M^(-1) ) x (YUV - K)
//          1   ~0              1.401999
//  M^-1 =  1   -0.344136       -0.714136
//          1   1.772           ~0
//
void Image::yuv2rgb()
{       
        if(_colorSpace == colorSpace::yuv)
        {
                int numPixels = _rows*_cols*3;
                unsigned char r = 0, g = 0, b = 0, y =0, u = 0, v =0;

                for (int i = 0; i < numPixels; i+=3)
                {
                        y = _pixels[i];
                        u = _pixels[i+1];
                        v = _pixels[i+2];

                        // Make sure all values are in the range [0,255]
                        r = (unsigned char)clamp(1*y + 1.401999*(v -128));
                        g = (unsigned char)clamp(1*y  - 0.344136*(u -128) - 0.714136*(v -128 ));
                        b = (unsigned char)clamp(1*y + 1.772*(u -128));

                        _pixels[i]= r;
                        _pixels[i+1] = g;
                        _pixels[i+ 2] = b;
                }

                _colorSpace = colorSpace::rgb;
        }
}

// Parse a string in a file to an integer untill a whitespace (ASCII Code 32) or a newline-character is found
int parseNumber(FILE * file)
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
header getHeader(FILE* src)
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
void imageToArray (int rows, int cols, int channels, fileType type, FILE * src, unsigned char* dst)
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

/*char parseChar(FILE * file)
{
     char current_char = fgetc(file);
     char number = 0;
     string comment;
    do
    {
        // Comments begin with a '#' (ASCII Code 35); the rest of the line should be ignored
        if(current_char == '#')
        {
           while ( fgetc(file)!='\n')
           {
                fgetc(file);
           }  
        }

        // The first character might be a whitespace; in this case it should be ignored
        else if(current_char != 32 && current_char != '\n')
        {
            number *= 10;
            // "Padding" ASCII-Decimal code to figures (0-9)
            number += (current_char - 48);
        }

        current_char = fgetc(file) ;
    }
    while(current_char != 32 && current_char != '\n');
    return number;
}*/

double clamp(double x, double min, double max)
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

int clamp(int x, int min, int max)
{
        int y = x;

        if(x<min)
        {
                y = min;
        }
        else if(x>=max)
        {
                //In order to avoid artifacts because of double to char conversion
                y = max;
        }

        return y;
}

