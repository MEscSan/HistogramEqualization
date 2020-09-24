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
    Image lennaPPM("../../TestImages/Lenna.ppm");
    Image lennaColor2Gvp("../../TestImages/Lenna.ppm");
    
    cout << "Beginning GPU Test\n";

    lennaColor2Gvp.color2gvp();
    lennaColor2Gvp.save("../../TestImages/LennaGvpCuda");

    for(int i =0; i<10; i++)
    {
        lennaPPM.rgb2yuv();
        lennaPPM.yuv2rgb();
        Histogram lennaHistogram(lennaColor2Gvp);
        if(i == 0)
        {
            lennaHistogram.save("../../TestImages/LennaGvpHistogramCuda");
        }
        cout << "Image " << i+1 << " out of 10\n";
    }

    lennaPPM.save("../../TestImages/LennaRgbCuda");
    cout << "Beginning CPU Test\n";

    for(int i =0; i<10; i++)
    {
        lennaPPM.host_rgb2yuv();
        lennaPPM.host_yuv2rgb();
        Histogram lennaHistogram(lennaColor2Gvp, 1);
        if(i == 0)
        {
            lennaHistogram.save("../../TestImages/LennaGvpHistogramCPU");
        }
        cout << "Image " << i+1 << " out of 10\n";
    }

    return EXIT_SUCCESS;
}
