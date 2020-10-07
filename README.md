# Histogram equalization in Cuda
Cuda programming project for parallel computing
* Small library for .pbm, .ppm, .pgm handling
* Histogram, Histogram normalisation and histogram equalisation on both c++ and Cuda\n
* Default Benchmarking files (21 RGB and 21 Grey Value pictures)
* Benchmarking can be carried out with the HistogramBenchmarking-Programm. Command line call:
```bash
 ./HistogramBenchmarking [cpu_test] [inputImgPath]
	[cpu_test]: 
     		0 => algorithms run only on the gpu
     		else => algorithms run both on gpu (parallel) and cpu (sequential)
	[inputImgPath]: path to the test image (if no path is given, the Benchmark folder found in this repository is used)
```
* The image processing algorithms can be tested with the EqualizationTest-Programm. Command line call: 
```bash
 ./EqualizationTest [cpu_test] [inputImgPath]
	[cpu_test]: 
     		0 => algorithms run only on the gpu
     		else => algorithms run both on gpu (parallel) and cpu (sequential)
	[inputImgPath]: path to the test image (if no path is given, the Benchmark folder found in this repository is used)
```

Project created and tested on Ubuntu 20.04. and Ubuntu 18.04.
