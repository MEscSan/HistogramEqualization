#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CudaErrorHelper.h"
#include "CudaImageTools.h"
#include "CudaHistogramTools.h"


using namespace std;

int main()
{
    cout << "GPU Test: ";

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    clock_t start_t, end_t;
    double total_t;

    start_t = clock();
   // for(int j = 0; j<20; j++)
    //{
        for (int i = 1; i < 19; i++)
        {   
            string path = "../../TestImages/Benchmark/"+ std::to_string(i) +".ppm";
            Image test(path.data());
            Image testR = test.getChannel(0);
            Image testG = test.getChannel(1);
            Image testB = test.getChannel(2);
            Image testMultichannel(test.getRows(), test.getCols(), colorSpace::rgb, 255, fileType::ppmBin);
            Histogram hist = Histogram(test);
            Histogram histR = Histogram(testR);
            Histogram histG = Histogram(testG);
            Histogram histB = Histogram(testB); 
         
            hist.normalize();
            histR.normalize();
            histG.normalize();
            histB.normalize();
            path = "../../TestImages/Benchmark/Cuda"+ std::to_string(i) +"Normalized";
            test.save(path.data());
            path = "../../TestImages/Benchmark/Cuda"+ std::to_string(i) +"Normalized_RGB";
            testMultichannel.setChannel(testR, 0);
            testMultichannel.setChannel(testG, 1);
            testMultichannel.setChannel(testB, 2);
            testMultichannel.save(path.data());
         
            hist.equalize();
            histR.equalize();
            histG.equalize();
            histB.equalize();
            path = "../../TestImages/Benchmark/Cuda"+ std::to_string(i) +"Equalized";    
            test.save(path.data());
            path = "../../TestImages/Benchmark/Cuda"+ std::to_string(i) +"Equalized_RGB";
            testMultichannel.setChannel(testR, 0);
            testMultichannel.setChannel(testG, 1);
            testMultichannel.setChannel(testB, 2);
            testMultichannel.save(path.data());

            /*test.rgb2yuv();
            test.yuv2rgb();
            test.rgb2hsv();*/
      }
    //}
    end_t = clock();
    total_t = ((double)end_t - (double)start_t)/CLOCKS_PER_SEC;

    cout << total_t<< '\n';

    cout << "CPU Test: ";
    start_t = clock();
    //for(int j = 0; j<20; j++)
    //{
        for (int i = 1; i < 19; i++)
        {   
            string path = "../../TestImages/Benchmark/"+ std::to_string(i) +".ppm";
            Image test(path.data());
            Histogram hist = Histogram(test,1); 
            hist.host_normalize();    
            path = "../../TestImages/Benmark/Cpu"+ std::to_string(i) +"Normalized";
            //test.save(path.data());
            hist.host_equalize();
            path = "../../TestImages/Benchmark/Cpu"+ std::to_string(i) +"Equalized";    
            //test.save(path.data());
        }
    //}
    end_t = clock();
    total_t = ((double)end_t - (double)start_t)/CLOCKS_PER_SEC;

    cout << total_t<< '\n';

    /*
    cudaEventElapsedTime(&milliseconds, start, stop);

    lennaColor2Gvp.color2gvp();
    lennaColor2Gvp.save("../../TestImages/LennaColor2GvpCuda");

    for(int i =0; i<10; i++)
    {
        Image lennaRgb("../../TestImages/Lenna.ppm");
        Image testA("../../TestImages/TestAbig.ppm");
        Image lennaColor2Gvp("../../TestImages/Lenna.ppm");
        Image lennaGvp("../../TestImages/Lenna.pgm");
        
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

    //cudaEventRecord(start);

    start_t = clock();
    for(int i=0; i<1; i++)
    {
        lennaRgb.rgb2yuv();
        lennaRgb.yuv2rgb();
        Histogram lennaRgbHistogram(lennaRgb); 
        Histogram testAHistogram(testA);
        testAHistogram.normalize();
        lennaRgbHistogram.normalize();
        lennaRgb.save("../../TestImages/LennaRgbCudaNormalized");
        testA.save("../../TestImages/TestAbigCudaNormalized");
        lennaRgbHistogram.equalize();
        testAHistogram.equalize();
        lennaRgb.save("../../TestImages/LennaRgbCudaEqualized");
        testA.save("../../TestImages/TestAbigCudaEqualized");
    //}
    end_t = clock();
    total_t = ((double)end_t - (double)start_t)/CLOCKS_PER_SEC;
	//cudaEventElapsedTime(&milliseconds, start, stop);

    cout << total_t<< '\n';

    cout << "CPU Test: ";

    Image lennaGvpCPU("../../TestImages/Lenna.pgm");
    
    cudaEventRecord(start);
 
    start_t = clock();
    for(int i=0; i<1; i++)
    {
        lennaRgb.host_rgb2yuv();
        lennaRgb.host_yuv2rgb();
        Histogram lennaRgbHistogram(lennaRgb,1); 
        Histogram testAHistogram(testA,1);
        testAHistogram.host_normalize();
        lennaRgbHistogram.host_normalize();
        lennaRgb.save("../../TestImages/LennaRgbCpuNormalized");
        testA.save("../../TestImages/TestAbigCpuNormalized");
        lennaRgbHistogram.host_equalize();
        testAHistogram.host_equalize();
        lennaRgb.save("../../TestImages/LennaRgbCpuEqualized");
        testA.save("../../TestImages/TestAbigCpuEqualized");
    //}
    end_t = clock();
    total_t = ((double)end_t - (double)start_t)/CLOCKS_PER_SEC;
    cudaEventSynchronize(stop);
    
    //cudaEventElapsedTime(&milliseconds, start, stop);
    cout << total_t<< '\n';

    cudaEventDestroy(start);
	cudaEventDestroy(stop);*/
    return EXIT_SUCCESS;
}
