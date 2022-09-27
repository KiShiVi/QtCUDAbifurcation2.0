#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <atomic>


enum BifurcationMode
{
	PARAM_A,
	PARAM_B,
	PARAM_C,
	SYMMETRY
};


__global__ void bifuractionKernel(//float* in_paramValues,
								  float*					in_paramA,
								  float*					in_paramB,
								  float*					in_paramC,
								  float*					in_symmetry,
								  int						in_nPts,
								  int						in_TMax,
								  float						in_h,
								  float						in_initialCondition1,
								  float						in_initialCondition2,
								  float						in_initialCondition3,
								  int						in_nValue,
								  float						in_prePeakFinderSliceK,
								  float*					in_data,
								  int*						in_dataSizes);
								  //float						in_A,
								  //float						in_B,
								  //float						in_C,
								  //int						in_mode);

// Обертка, которая для каждой бифуркационки будет своя!
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


__device__ void peakFinder(	int idx, 
							float prePeakFinder, 
							size_t amountOfTPoints, 
							float* in_data, 
							int* out_dataSizes, 
							float* out_data);

// return in "out" array [a, b] with "amount" elements
template <class T1, class T2>
__host__ void linspace(T1 a, T1 b, int amount, T2* out);

__host__ void getParamsAndSymmetry(float* a, float* b, float* c, float* symmetry,
	float in_a, float in_b, float in_c, float in_symmetry,
	float startInterval1, float finishInteraval1, int mode1,
	int nPts);

template <class T>
__host__ void slice(T* in, int a, int b, T* out);