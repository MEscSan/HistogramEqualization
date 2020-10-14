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

// Equalizes and Normalizes becnhmark-images using the methods and classes implemented in the project and saves the resulting images
// Users can decide whether to test he sequential (cpu) version of the algorithms 
// Users can give their own benchmark-image path
//
// Command line call:
// ./EqualizationTest [cpu_test] [inputImgPath]
// [cpu_test]: 
//      0 => algorithms run only on the gpu
//      else => algorithms run both on gpu(parallel) and cpu(sequential)
// [inputImgPath]: path to the test image
//
// Remark:  gvp...grey-value-picture (in code and comments) 

// Normalizes and equalizes an input color image both with RGB<->YCbCr conversion and processing the R-, G- and B-chanels separately
// Algorithms un sequentially on the cpu
void CPU_RGB_HistogramOperationsTest(string path)
{
    // Initialize Image object with input picture
    Image test(path.data());

    // Get R-, G- and B- channels from test image
    Image testR = test.getChannel(0);
    Image testG = test.getChannel(1);
    Image testB = test.getChannel(2);    

    // Initialize histograms from all 4 images on the cpu
    Histogram hist=Histogram(test,1);
    Histogram histR=Histogram(testR,1);
    Histogram histG=Histogram(testG,1);
    Histogram histB=Histogram(testB,1);

    // Normalize all histograms sequentially on the host machine
    hist.host_normalize();
    histR.host_normalize();
    histG.host_normalize();
    histB.host_normalize();

    // Initialize image object to contain the separately processed R-, G- and B-Channels
    Image testMultichannel(test.getRows(), test.getCols(), colorSpace::rgb, 255, fileType::ppmBin);

    // Save the processed images into the same directory as the imput image
    string savePath = path.substr(0, path.length() - 4) +"CPU_Normalized";
    test.save(savePath.data());

    savePath = path.substr(0, path.length() - 4) + "CPU_Normalized_RGB";

    // Insert separately processed R-, G- and B-Channels into testMultichannel-image and save it
    testMultichannel.setChannel(testR, 0);
    testMultichannel.setChannel(testG, 1);
    testMultichannel.setChannel(testB, 2);
    testMultichannel.save(savePath.data());
    
    // Equalize all histograms sequentially on the host machine
    hist.host_equalize();
    histR.host_equalize();
    histG.host_equalize();
    histB.host_equalize();

    // Save YCbCr-equalized image into the same directory as the imput image
    savePath = path.substr(0, path.length() - 4) + "CPU_Equalized";    
    test.save(savePath.data());

    // Insert equalized R-, G- and B-Channels into testMultichannel-image and save it
    savePath = path.substr(0, path.length() - 4) +"CPU_Equalized_RGB";
    testMultichannel.setChannel(testR, 0);
    testMultichannel.setChannel(testG, 1);
    testMultichannel.setChannel(testB, 2);
    testMultichannel.save(savePath.data());
}

// Normalizes and equalizes an input grey value picture
// ALgorithms run sequentially on the cpu
void CPU_GVP_HistogramOperationsTest(string path)
{
    // Initialize Image object with input gvp
    Image test(path.data());

    // Initialize histogram of input image on the cpu 
    Histogram hist = Histogram(test,1);
  
    // Normalize histogram sequentially on the cpu and save it into the same directory as the imput image
    hist.host_normalize();
    string savePath = path.substr(0, path.length() - 4) +"CPU_Normalized";
    test.save(savePath.data());

    // Equalize image and histogram sequentially on the cpu and save it into the same directory as the imput image
    hist.host_equalize();
    savePath = path.substr(0, path.length() - 4) + "CPU_Equalized";    
    test.save(savePath.data());
}

// Normalizes and equalizes an input color image both with RGB<->YCbCr conversion and processing the R-, G- and B-chanels separately
// Algorithms run in parallel on the gpu
void GPU_RGB_HistogramOperationsTest(string path)
{
    // Initialize image object with input picture
    Image test(path.data());
    
    // Get R-, G- and B-channels from input picture
    Image testR = test.getChannel(0);
    Image testG = test.getChannel(1);
    Image testB = test.getChannel(2);

    // Initialize image object to contain the separately processed R-, G- and B-Channels
    Image testMultichannel(test.getRows(), test.getCols(), colorSpace::rgb, 255, fileType::ppmBin);

    // Initialize histograms on the CUDA device
    Histogram hist=Histogram(test);
    Histogram histR=Histogram(testR);
    Histogram histG=Histogram(testG);
    Histogram histB=Histogram(testB);

    // Normalize histograms on the CUDA device
    hist.dev_normalize();
    histR.dev_normalize();
    histG.dev_normalize();
    histB.dev_normalize();
    
    // Save the YCbCr-normalized image into the same directory as the imput image
    string savePath = path.substr(0, path.length() - 4)+"CUDA_Normalized";
    test.save(savePath);

    // Insert normalized R-, G- and B-Channels into testMultichannel-image and save it
    savePath = path.substr(0, path.length() - 4) + "CUDA_Normalized_RGB";
    testMultichannel.setChannel(testR, 0);
    testMultichannel.setChannel(testG, 1);
    testMultichannel.setChannel(testB, 2);
    testMultichannel.save(savePath);
    
    // Equalize histograms on the CUDA device
    hist.dev_equalize();
    histR.dev_equalize();
    histG.dev_equalize();
    histB.dev_equalize();

    // Save the YCbCr-equalized image into the same directory as the imput image
    savePath = path.substr(0, path.length() - 4) +"CUDA_Equalized";    
    test.save(savePath.data());

    // Insert equalized R-, G- and B-Channels into testMultichannel-image and save it into the same directory as the imput image
    savePath = path.substr(0, path.length() - 4)  +"CUDA_Equalized_RGB";
    testMultichannel.setChannel(testR, 0);
    testMultichannel.setChannel(testG, 1);
    testMultichannel.setChannel(testB, 2);
    testMultichannel.save(savePath.data());
}

// Normalizes and equalizes an input gvp
// Algorithms run in parallel on the gpu
void GPU_GVP_HistogramOperationsTest(string path)
{
    // Initialize image object with input gvp
    Image test(path.data());

    // Initialize histogram of input picture using the CUDA device
    Histogram hist = Histogram(test);

    // Normalize histogram of input picture on CUDA device
    hist.dev_normalize();
    
    // Save normalized image into the same directory as the imput image
    string savePath = path.substr(0, path.length() - 4)+"CUDA_Normalized";
    test.save(savePath.data());
    
    // Equalize image and histogram on the CUDA device
    hist.dev_equalize();

    //Save the equalized gvp into the same directory as the imput image
    savePath = path.substr(0, path.length() - 4) +"CUDA_Equalized";    
    test.save(savePath.data());

}

int main(int argc, char* argv[] )
{
    // Initialize image path variable with default image path
    // Default test images found on Benchmark directory (see repository)
    string inputImgPath = "../../Benchmark/";
    
    // Initialize image extension variable, default test images are both RGB-Images (.ppm) and GVPs (.pgm)
    string inputImgExtension = "ppm_pgm";

    // Get path for benchmark-image from command line if given
    if(argc > 2)
    {
        inputImgPath = argv[2];
        inputImgExtension = inputImgPath.substr(inputImgPath.length() - 3, 3);
    }    
    
    // Check if the first input parameter is unequal 0, if it is, run all algorithms on the cpu
    if(argc > 1 && atoi(argv[1])!= 0)
    {
        // If the input image is an RGB-image
        if(inputImgExtension == "ppm")
        {
            CPU_RGB_HistogramOperationsTest(inputImgPath);
        }
        // If the input image is a grey-value picture
        else if(inputImgExtension == "pgm")
        {
            CPU_GVP_HistogramOperationsTest(inputImgPath);
        }
        // If no image path given, use default path
        else
        {
            for(int i = 0; i < 21; i++ )
            {
                string defaultPath = inputImgPath + std::to_string(i) +".ppm";
                CPU_RGB_HistogramOperationsTest(defaultPath);

                defaultPath = inputImgPath + std::to_string(i) +".pgm";
                CPU_GVP_HistogramOperationsTest(defaultPath);
            }
        }
    }

    // Run parallel algorithms on the gpu
    if(inputImgExtension == "ppm")
    {
        GPU_RGB_HistogramOperationsTest(inputImgPath);
    }
    else if(inputImgExtension == "pgm")
    {
        GPU_GVP_HistogramOperationsTest(inputImgPath);
    }
    else
    {
        for(int i = 0; i < 21; i++ )
        {
            string rgbDefaultPath = inputImgPath + std::to_string(i) +".ppm";
            GPU_RGB_HistogramOperationsTest(rgbDefaultPath);

            string gvpDefaultPath = inputImgPath + std::to_string(i) +".pgm";
            GPU_GVP_HistogramOperationsTest(gvpDefaultPath);
        }
    }

    return EXIT_SUCCESS;
}
