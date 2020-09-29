/*
    Small library with a minimal class and methods required for manipulating, loading and saving images in netpbm formats (.pbm, .pgm, .ppm)
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
    yuv
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
	    ->pixels: pointer to a one-dimensional byte-array containing the pixel-values of the image as stored in the file
	    ->number of Values: number of values a single pixel can take (should be between 1 and 255)
        ->file: image file from where to extract the information
        ->color space
	    */

        int _rows;
        int _cols;
        int _channels;
        fileType _type;
        unsigned char* _pixels;
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
        void color2gvp();
        void gvp2color();
        void rgb2yuv();
        void yuv2rgb();
};

// Functions used inside the Image-methods

// Parse a string in a file to an integer till a whitespace (ASCII Code 32), end of line ('\n') or a comment is found
// Comments a strings beginning with '#' and ending with a '\n' character
int parseNumber(FILE * file);
//char parseChar(FILE*file);

// Returns the header-struct of the image source file
header getHeader(FILE* src);

// Reads the pixel values from the original file and stores them in the image-object as a byte-array 
// WARNING: The method assumes that the file pointer points to the first pixel, getHeader must be called before
void imageToArray(int rows, int cols, int channels, fileType type, FILE * src, unsigned char* dst);

// Keeps the given value x within the boundaries [min, max](used mainly in the double-to-byte conversion)  
double clamp(double x, double min = 0, double max = 255);
int clamp (int x, int min = 0, int max = 255);
