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
	LORENZ_RYBIN,
	DEFF
};


__device__ void calculateDiscreteModel(int mode, double* x, double* values, double h);

__global__ void bifuractionKernel(	int in_nPts,
									int in_TMax,
									double in_h,
									double* in_initialConditions,
									int in_nValue,
									double in_prePeakFinderSliceK,
									float* in_data,
									int* in_dataSizes,
									ResultMode resultMode,
									int thresholdValueOfMaxSignalValue,
									int	in_amountOfParams,
									int in_discreteModelMode,
									int	in_prescaller,
									double* in_params,
									double* in_paramValues1,
									int	in_mode1,
									double* in_paramValues2 = nullptr,
									int in_mode2 = -1,
									double* in_paramValues3 = nullptr,
									int in_mode3 = -1,
									int in_kdeSampling = 0,
									double in_kdeSamplesInterval1 = 0.0f,
									double in_kdeSamplesInterval2 = 0.0f,
									double in_kdeSmoothH = 0.0f);


// Обертка, которая для каждой бифуркационки будет своя!
__host__ void bifurcation1D(float					in_tMax,
							int					in_nPts,
							double				in_h,
							double*				in_initialConditions,
							double				in_paramValues1,
							double				in_paramValues2,
							int					in_nValue,
							double				in_prePeakFinderSliceK,
							int					in_thresholdValueOfMaxSignalValue,
							int					in_amountOfParams,
							int					in_discreteModelMode,
							int					in_prescaller,
							double*				in_params,
							int					in_mode,
							double				in_memoryLimit,
							std::string			in_outPath,
							bool				in_debug,
							std::atomic<int>& progress);



// Обертка, которая для каждой бифуркационки будет своя!
__host__ void bifurcation2D(float					in_tMax,
	int					in_nPts,
	double				in_h,
	double*				in_initialConditions,
	double				in_paramValues1,
	double				in_paramValues2,
	double				in_paramValues3,
	double				in_paramValues4,
	int					in_nValue,
	double				in_prePeakFinderSliceK,
	int					in_thresholdValueOfMaxSignalValue,
	int					in_amountOfParams,
	int					in_discreteModelMode,
	int					in_prescaller,
	double*				in_params,
	int					in_mode1,
	int					in_mode2,
	int					in_kdeSampling,
	double				in_kdeSamplesInterval1,
	double				in_kdeSamplesInterval2,
	double				in_kdeSamplesSmooth,
	double				in_memoryLimit,
	std::string			in_outPath,
	bool				in_debug,
	std::atomic<int>& progress);



// Обертка, которая для каждой бифуркационки будет своя!
__host__ void bifurcation3D(float					in_tMax,
	int					in_nPts,
	double				in_h,
	double*				in_initialConditions,
	double				in_paramValues1,
	double				in_paramValues2,
	double				in_paramValues3,
	double				in_paramValues4,
	double				in_paramValues5,
	double				in_paramValues6,
	int					in_nValue,
	double				in_prePeakFinderSliceK,
	int					in_thresholdValueOfMaxSignalValue,
	int					in_amountOfParams,
	int					in_discreteModelMode,
	int					in_prescaller,
	double*				in_params,
	int					in_mode1,
	int					in_mode2,
	int					in_mode3,
	int					in_kdeSampling,
	double				in_kdeSamplesInterval1,
	double				in_kdeSamplesInterval2,
	double				in_kdeSamplesSmooth,
	double				in_memoryLimit,
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

__host__ void getParamsAndSymmetry2D(double* param1, double* param2,
	double startInterval1, double finishInteraval1,
	double startInterval2, double finishInteraval2,
	int nPts);

__host__ void getParamsAndSymmetry3D(double* param1, double* param2, double* param3,
	double startInterval1, double finishInteraval1,
	double startInterval2, double finishInteraval2,
	double startInterval3, double finishInteraval3,
	int nPts);

template <class T>
__host__ void slice(T* in, int a, int b, T* out);