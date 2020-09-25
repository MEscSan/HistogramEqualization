#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CudaErrorHelper.h"
#include "CudaImageTools.h"
#include "CudaHistogramTools.h"

using namespace std;

int main()
{
    Image lennaRgb("../../TestImages/Lenna.ppm");
    Image lennaColor2Gvp("../../TestImages/Lenna.ppm");
    Image lennaGvp("../../TestImages/Lenna.pgm");
    
    cout << "Beginning GPU Test\n";

    lennaColor2Gvp.color2gvp();
    lennaColor2Gvp.save("../../TestImages/LennaColor2GvpCuda");

    for(int i =0; i<10; i++)
    {
        lennaRgb.rgb2yuv();
        lennaRgb.yuv2rgb();
        Histogram lennaGvpHistogram(lennaGvp);
        Histogram lennaRgbHistogram(lennaRgb);   
        if(i == 0)
        {
            lennaGvpHistogram.save("../../TestImages/LennaGvpHistogramCuda"); 
            lennaRgbHistogram.save("../../TestImages/LennaRgbHistogramCuda");  
            lennaRgb.save("../../TestImages/LennaRgbCuda");
            lennaGvp.save("../../TestImages/LennaGvpCuda");
            lennaGvpHistogram.normalize();
            lennaRgbHistogram.normalize();
            lennaGvpHistogram.save("../../TestImages/LennaGvpHistogramNormalizedCuda"); 
            lennaRgbHistogram.save("../../TestImages/LennaRgbHistogramNormalizedCuda"); 
            lennaRgb.save("../../TestImages/LennaRgbCudaNormalized");
            lennaGvp.save("../../TestImages/LennaGvpCudaNormalized");
            lennaGvpHistogram.equalize();
            lennaRgbHistogram.equalize();
            lennaGvpHistogram.save("../../TestImages/LennaGvpHistogramEqualizedCuda"); 
            lennaRgbHistogram.save("../../TestImages/LennaRgbHistogramEqualizedCuda"); 
            lennaRgb.save("../../TestImages/LennaRgbCudaEqualized");
            lennaGvp.save("../../TestImages/LennaGvpCudaEqualized");
        }
        
        cout << "Image " << i+1 << " out of 10\n";
    }

    cout << "Beginning CPU Test\n";

    Image lennaGvpCPU("../../TestImages/Lenna.pgm");
    
    for(int i =0; i<10; i++)
    {
        lennaRgb.host_rgb2yuv();
        lennaRgb.host_yuv2rgb();
        Histogram lennaHistogram(lennaGvpCPU, 1);
        if(i == 0)
        {
            lennaHistogram.save("../../TestImages/LennaGvpHistogramCPU");
            lennaHistogram.host_normalize();
            lennaHistogram.save("../../TestImages/LennaGvpNormalizedHistogramCPU");
            lennaHistogram.host_equalize();
            lennaHistogram.save("../../TestImages/LennaGvpEqualizedHistogramCPU");
            
        }


        cout << "Image " << i+1 << " out of 10\n";
    }

    return EXIT_SUCCESS;
}
