#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <atomic>

enum ResultMode
{
	PEAKFINDER_MODE,
	KDE_MODE
};

enum BifurcationMode
{
	PARAM_A,
	PARAM_B,
	PARAM_C,
	SYMMETRY
};


__global__ void bifuractionKernel(	float*					in_paramA,
									float*					in_paramB,
									float*					in_paramC,
									float*					in_symmetry,
									int						in_nPts,
									int						in_TMax,
									float					in_h,
									float					in_initialCondition1,
									float					in_initialCondition2,
									float					in_initialCondition3,
									int						in_nValue,
									float					in_prePeakFinderSliceK,
									float*					in_data,
									int*					in_dataSizes,
									ResultMode				resultMode,
									int						in_kdeSampling = 0,
									float					in_kdeSamplesInterval1 = 0,
									float					in_kdeSamplesInterval2 = 0,
									float					in_kdeSmoothH = 0);


// �������, ������� ��� ������ ������������� ����� ����!
__host__ void bifurcation1D(int					in_tMax,
	int					in_nPts,
	float				in_h,
	float				in_initialCondition1,
	float				in_initialCondition2,
	float				in_initialCondition3,
	float				in_paramValues1,
	float				in_paramValues2,
	int					in_nValue,
	float				in_prePeakFinderSliceK,
	float				in_paramA,
	float				in_paramB,
	float				in_paramC,
	int					in_mode,
	float				in_memoryLimit,
	std::string			in_outPath,
	bool				in_debug,
	std::atomic<int>&	progress);



// �������, ������� ��� ������ ������������� ����� ����!
__host__ void bifurcation2D(int					in_tMax,
	int					in_nPts,
	float				in_h,
	float				in_initialCondition1,
	float				in_initialCondition2,
	float				in_initialCondition3,
	float				in_paramValues1,
	float				in_paramValues2,
	float				in_paramValues3,
	float				in_paramValues4,
	int					in_nValue,
	float				in_prePeakFinderSliceK,
	float				in_paramA,
	float				in_paramB,
	float				in_paramC,
	int					in_mode1,
	int					in_mode2,
	int					in_kdeSampling,
	float				in_kdeSamplesInterval1,
	float				in_kdeSamplesInterval2,
	float				in_kdeSamplesSmooth,
	float				in_memoryLimit,
	std::string			in_outPath,
	bool				in_debug,
	std::atomic<int>& progress);



__device__ int peakFinder(	int idx, 
							float prePeakFinder, 
							size_t amountOfTPoints, 
							float* in_data, 
							int* out_dataSizes, 
							float* out_data);



__device__ void kdeMethod(int idx,
	float* data,
	int* kdeResult,
	int kdeSampling,
	int _outSize,
	float kdeSamplesInterval1,
	float kdeSamplesInterval2,
	size_t amountOfTPoints,
	float kdeSmoothH);



// return in "out" array [a, b] with "amount" elements
template <class T1, class T2>
__host__ void linspace(T1 a, T1 b, int amount, T2* out, int startIndex = 0);

__host__ void getParamsAndSymmetry(float* a, float* b, float* c, float* symmetry,
	float in_a, float in_b, float in_c, float in_symmetry,
	float startInterval1, float finishInteraval1, int mode1,
	int nPts);

__host__ void getParamsAndSymmetry(float* a, float* b, float* c, float* symmetry,
	float in_a, float in_b, float in_c, float in_symmetry,
	float startInterval1, float finishInteraval1, int mode1,
	float startInterval2, float finishInteraval2, int mode2,
	int nPts);

template <class T>
__host__ void slice(T* in, int a, int b, T* out);