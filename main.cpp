#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>

//#include "CudaErrorHelper.h"
#include "ImageTools.h"
#include "HistogramTools.h"

using namespace std;

int main()
{
    Image lennaPPM("../../TestImages/Lenna.ppm");
    Image lennaPGM("../../TestImages/Lenna.pgm");
    Image landscape("../../TestImages/Landscape.pgm");
    Image lennaPBM("../../TestImages/Lenna.pbm");
    Image random(1080,1080,colorSpace::rgb,255,fileType::ppmBin);
    Image random2(512,1024,colorSpace::rgb,1,fileType::pbmASCII);

    Image colors("../../TestImages/colors.ppm");
    Image feep("../../TestImages/feep.pgm");
    Image j("../../TestImages/j.pbm");
    
    colors.save("../../TestImages/colorsFromObject");
    feep.save("../../TestImages/feepFromObject");
    j.save("../../TestImages/jFromObject");

    lennaPPM.save("../../TestImages/LennaFromObject");
    lennaPGM.save("../../TestImages/LennaFromObject");
    lennaPBM.save("../../TestImages/LennaFromObject");
    random.save("../../TestImages/Random");
    random2.save("../../TestImages/Random2");

    lennaPPM.rgb2yuv();
    lennaPPM.save("../../TestImages/LennaYUV");    

    lennaPPM.yuv2rgb();
    lennaPPM.save("../../TestImages/LennaRGB"); 

    Histogram histLennaRGB = Histogram(lennaPPM); 
    histLennaRGB.save("../../TestImages/HistogramLennaRGB");
    histLennaRGB.normalize();    
    histLennaRGB.save("../../TestImages/HistogramLennaRGBNormalized");
    lennaPPM.save("../../TestImages/LennaRGBNormalized");
    histLennaRGB.equalize();
    lennaPPM.save("../../TestImages/LennaRGBEqualized");
    

    Image landscapePPM("../../TestImages/Landscape.ppm");
    Histogram histLandscapeRGB = Histogram(landscapePPM); 
    histLandscapeRGB.normalize();    
    landscapePPM.save("../../TestImages/LandscapeRGBNormalized");

    Image br180("../../TestImages/br180.ppm");
    Histogram histBR180 = Histogram(br180); 
    histBR180.normalize();    
    br180.save("../../TestImages/BR180Normalized");
    histBR180.equalize();    
    br180.save("../../TestImages/BR180Equalized");

    Image br155("../../TestImages/br155.ppm");
    Histogram histBR155 = Histogram(br155); 
    histBR155.normalize();    
    br155.save("../../TestImages/BR155Normalized");
    histBR155.equalize();    
    br155.save("../../TestImages/BR155Equalized");

    for (int i = 1; i < 10; i++)
    {   
        string path = "../../TestImages/Test"+ std::to_string(i) +".ppm";
        Image test(path.data());
        Histogram hist = Histogram(test); 
        hist.normalize();    
        path = "../../TestImages/Test"+ std::to_string(i) +"Normalized";
        test.save(path.data());
        hist.equalize();
        path = "../../TestImages/Test"+ std::to_string(i) +"Equalized";    
        test.save(path.data());
    }
    
        /*Image test10("../../TestImages/TestA.ppm");
        Histogram hist = Histogram(test10); 
        hist.normalize();    
        string path = "../../TestImages/Test10Normalized";
        test10.save(path.data());
        hist.equalize();
        path = "../../TestImages/Test10Equalized";    
        test10.save(path.data()); 
        
        */

            
        Image test11("../../TestImages/TestB.pgm");
        Histogram hist11 = Histogram(test11); 
        hist11.normalize();    
        string path = "../../TestImages/TestNormalized";
        test11.save(path.data());
        hist11.equalize();
        path = "../../TestImages/TestEqualized";    
        test11.save(path.data());

    //histLennaRGB.equalize();
    //lennaPPM.save("../../TestImages/LennaRGBEqualized");
    Image test4("../../TestImages/Test4.ppm");
    test4.rgb2yuv();
    Image y = test4.getChannel(0);
    y.save("../../TestImages/LennaYChannel");
    Image u = test4.getChannel(1);
    u.save("../../TestImages/LennaUChannel");
    Image v = test4.getChannel(2);
    v.save("../../TestImages/LennaVChannel");


    Histogram histLennaPGM = Histogram(lennaPGM);
    histLennaPGM.save("../../TestImages/HistogramLena");
    histLennaPGM.normalize();
    histLennaPGM.save("../../TestImages/HistogramLenaNormalized");
    lennaPGM.save("../../TestImages/LennaNormalized");
    histLennaPGM.equalize();
    histLennaPGM.save("../../TestImages/HistogrammLennaEqualized");
    lennaPGM.save("../../TestImages/LennaEqualized");

    Histogram histLandscape = Histogram(landscape);
    histLandscape.save("../../TestImages/HistogramLandscape");
    histLandscape.normalize();
    landscape.save("../../TestImages/LandscapeNormalized");
    histLandscape.save("../../TestImages/HistogramLandscapeNormalized");
    histLandscape.equalize();
    histLandscape.save("../../TestImages/HistogramLandscapeEqualized");
    landscape.save("../../TestImages/LandscapeEqualized");


    return EXIT_SUCCESS;
}
