#pragma once 

#include <cstdlib>
#include <cstdio>
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include "ImageTools.h"

using namespace std;

//TODO: Implement RGB-handling
class Histogram
{
    private:
        int _numValues;
        int* _values;
        double* _valuesCumulative;
        unsigned char* _lookUpTable;
        unsigned char _minValue;
        unsigned char _maxValue;
        // In order for changes in source-image to be consequently applied, we need to pass the source by reference, and not by value
        // Otherwise we would be passing only the pixel information by reference, but not the image object itself
        Image& _src;

    public:
        Histogram(Image& src);

        void calculate();
        void display(ostream& output = cout);    
        void equalize();
        int* getHistogramPtr();
        void normalize();
        void save(string path);

};

int getMax(int* arrayPtr, int arraySize, colorSpace cs = colorSpace::gvp);
unsigned char getMax(unsigned char* arrayPtr, int arraySize, colorSpace cs = colorSpace::gvp);
