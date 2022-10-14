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

enum DISCRETE_MODEL
{	
	ROSSLER,
	CHEN,
	LORENZ,
	LORENZ_RYBIN
};


__device__ void calculateDiscreteModel(int mode, float* x, float* values, float h);

__global__ void bifuractionKernel(	int in_nPts,
									int in_TMax,
									float in_h,
									float* in_initialConditions,
									int in_nValue,
									float in_prePeakFinderSliceK,
									float* in_data,
									int* in_dataSizes,
									ResultMode resultMode,
									int thresholdValueOfMaxSignalValue,
									int	in_amountOfParams,
									int in_discreteModelMode,
									int	in_prescaller,
									float* in_params,
									float* in_paramValues1,
									int	in_mode1,
									float* in_paramValues2 = nullptr,
									int in_mode2 = -1,
									float* in_paramValues3 = nullptr,
									int in_mode3 = -1,
									int in_kdeSampling = 0,
									float in_kdeSamplesInterval1 = 0.0f,
									float in_kdeSamplesInterval2 = 0.0f,
									float in_kdeSmoothH = 0.0f);


// Обертка, которая для каждой бифуркационки будет своя!
__host__ void bifurcation1D(int					in_tMax,
							int					in_nPts,
							float				in_h,
							float*				in_initialConditions,
							float				in_paramValues1,
							float				in_paramValues2,
							int					in_nValue,
							float				in_prePeakFinderSliceK,
							int					in_thresholdValueOfMaxSignalValue,
							int					in_amountOfParams,
							int					in_discreteModelMode,
							int					in_prescaller,
							float*				in_params,
							int					in_mode,
							float				in_memoryLimit,
							std::string			in_outPath,
							bool				in_debug,
							std::atomic<int>& progress);



// Обертка, которая для каждой бифуркационки будет своя!
__host__ void bifurcation2D(int					in_tMax,
	int					in_nPts,
	float				in_h,
	float*				in_initialConditions,
	float				in_paramValues1,
	float				in_paramValues2,
	float				in_paramValues3,
	float				in_paramValues4,
	int					in_nValue,
	float				in_prePeakFinderSliceK,
	int					in_thresholdValueOfMaxSignalValue,
	int					in_amountOfParams,
	int					in_discreteModelMode,
	int					in_prescaller,
	float*				in_params,
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



// Обертка, которая для каждой бифуркационки будет своя!
__host__ void bifurcation3D(int					in_tMax,
	int					in_nPts,
	float				in_h,
	float*				in_initialConditions,
	float				in_paramValues1,
	float				in_paramValues2,
	float				in_paramValues3,
	float				in_paramValues4,
	float				in_paramValues5,
	float				in_paramValues6,
	int					in_nValue,
	float				in_prePeakFinderSliceK,
	int					in_thresholdValueOfMaxSignalValue,
	int					in_amountOfParams,
	int					in_discreteModelMode,
	int					in_prescaller,
	float*				in_params,
	int					in_mode1,
	int					in_mode2,
	int					in_mode3,
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
	float kdeSmoothH,
	int criticalValueOfPeaks);



// return in "out" array [a, b] with "amount" elements
template <class T1, class T2>
__host__ void linspace(T1 a, T1 b, int amount, T2* out, int startIndex = 0);

__host__ void getParamsAndSymmetry2D(float* param1, float* param2,
	float startInterval1, float finishInteraval1,
	float startInterval2, float finishInteraval2,
	int nPts);

__host__ void getParamsAndSymmetry3D(float* param1, float* param2, float* param3,
	float startInterval1, float finishInteraval1,
	float startInterval2, float finishInteraval2,
	float startInterval3, float finishInteraval3,
	int nPts);

template <class T>
__host__ void slice(T* in, int a, int b, T* out);