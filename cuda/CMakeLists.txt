cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(HISTOGRAM_EQUALIZATION VERSION 1.0 LANGUAGES CXX CUDA)

add_executable(EqualizationTest EqualizationTest.cu CudaImageTools.cu CudaHistogramTools.cu)
add_executable(HistogramBenchmarking HistogramBenchmarking.cu CudaImageTools.cu CudaHistogramTools.cu)

#Supress warning about deprecated architectures
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
	
# Version dependent flags for CUDA-Architectures
if(${CMAKE_VERSION} VERSION_LESS "3.18.4")
	#Set flags for compute-50, compute_52 and compute_53 virtual architecture
	set(CMAKE_CUDA_FLAGS " -gencode=arch=compute_50,code=sm_50, -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_53,code=sm_53  -Wno-deprecated-gpu-targets")
else()
	#Set flags for virtual and real CUDA-Architectures 50 (GeForce 840M), 52(Standard) and 53 (Tegra X1)
	set_property(TARGET EqualizationTest PROPERTY CUDA_ARCHITECTURES 50 52 53)
	set_property(TARGET HistogramBenchmarking PROPERTY CUDA_ARCHITECTURES 50 52 53)
endif()

target_link_libraries(HistogramBenchmarking stdc++fs)



