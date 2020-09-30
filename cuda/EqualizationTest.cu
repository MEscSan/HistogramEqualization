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


// Equalizes and Normalizes Becnhmark-Images using the methods and classes implemented in the project
int main()
{
    cout << "GPU Test: ";
    
    clock_t start_t, end_t;
    double total_t;

    start_t = clock();
 
    for (int i = 0; i < 21; i++)
    {   
        string path = "../Benchmark/"+ std::to_string(i) +".ppm";
        Image test(path.data());
        Image testR = test.getChannel(0);
        Image testG = test.getChannel(1);
        Image testB = test.getChannel(2);
        Image testMultichannel(test.getRows(), test.getCols(), colorSpace::rgb, 255, fileType::ppmBin);
        Histogram hist = Histogram(test);
        Histogram histR = Histogram(testR);
        Histogram histG = Histogram(testG);
        Histogram histB = Histogram(testB);
        hist.dev_normalize();
        histR.dev_normalize();
        histG.dev_normalize();
        histB.dev_normalize();
        
        path = "../Benchmark/Cuda"+ std::to_string(i) +"Normalized";
        test.save(path.data());
        path = "../Benchmark/Cuda"+ std::to_string(i) +"Normalized_RGB";
        testMultichannel.setChannel(testR, 0);
        testMultichannel.setChannel(testG, 1);
        testMultichannel.setChannel(testB, 2);
        testMultichannel.save(path.data());
        
        hist.dev_equalize();
        histR.dev_equalize();
        histG.dev_equalize();
        histB.dev_equalize();
        path = "../Benchmark/Cuda"+ std::to_string(i) +"Equalized";    
        test.save(path.data());
        path = "..//Benchmark/Cuda"+ std::to_string(i) +"Equalized_RGB";
        testMultichannel.setChannel(testR, 0);
        testMultichannel.setChannel(testG, 1);
        testMultichannel.setChannel(testB, 2);
        testMultichannel.save(path.data());

    }

    end_t = clock();
    total_t = ((double)end_t - (double)start_t)/CLOCKS_PER_SEC;

    cout << total_t<< '\n';

    cout << "CPU Test: ";
    start_t = clock();

    for (int i = 0; i < 21; i++)
    {   
        string path = "../Benchmark/"+ std::to_string(i) +".ppm";
        Image test(path.data());
        Image testR = test.getChannel(0);
        Image testG = test.getChannel(1);
        Image testB = test.getChannel(2);
        Image testMultichannel(test.getRows(), test.getCols(), colorSpace::rgb, 255, fileType::ppmBin);
        Histogram hist = Histogram(test,1);
        Histogram histR = Histogram(testR,1);
        Histogram histG = Histogram(testG,1);
        Histogram histB = Histogram(testB,1);
        hist.host_normalize();
        histR.host_normalize();
        histG.host_normalize();
        histB.host_normalize();
        
        path = "../Benchmark/Cpu"+ std::to_string(i) +"Normalized";
        test.save(path.data());
        path = "../Benchmark/Cpu"+ std::to_string(i) +"Normalized_RGB";
        testMultichannel.setChannel(testR, 0);
        testMultichannel.setChannel(testG, 1);
        testMultichannel.setChannel(testB, 2);
        testMultichannel.save(path.data());
        
        hist.host_equalize();
        histR.host_equalize();
        histG.host_equalize();
        histB.host_equalize();
        path = "../Benchmark/Cpu"+ std::to_string(i) +"Equalized";    
        test.save(path.data());
        path = "../Benchmark/Cpu"+ std::to_string(i) +"Equalized_RGB";
        testMultichannel.setChannel(testR, 0);
        testMultichannel.setChannel(testG, 1);
        testMultichannel.setChannel(testB, 2);
        testMultichannel.save(path.data());
    }
    end_t = clock();
    total_t = ((double)end_t - (double)start_t)/CLOCKS_PER_SEC;

    cout << total_t<< '\n';

    for (int i = 0; i < 21; i++)
    {   
        string path = "../Benchmark/"+ std::to_string(i) +".pgm";
        Image test(path.data());
        Histogram hist = Histogram(test);
        hist.dev_normalize();
        
        path = "../Benchmark/Cuda"+ std::to_string(i) +"Normalized";
        test.save(path.data());
        
        hist.dev_equalize();
        path = "../Benchmark/Cuda"+ std::to_string(i) +"Equalized";    
        test.save(path.data());
    }

    return EXIT_SUCCESS;
}
