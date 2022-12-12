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
	CONSERVA,
	DISSIPATA,
	TIMUR,
	CompCD,
	RK4
};


__device__ void calculateDiscreteModel(int mode, double* x, double* values, double h);

__global__ void bifuractionKernel(int in_nPts,
	double in_TMax,
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
	float in_kdeSamplesInterval1 = 0.0f,
	float in_kdeSamplesInterval2 = 0.0f,
	float in_kdeSmoothH = 0.0f);


// Обертка, которая для каждой бифуркационки будет своя!
__host__ void bifurcation1D(double					in_tMax,
	int					in_nPts,
	double				in_h,
	double* in_initialConditions,
	double				in_paramValues1,
	double				in_paramValues2,
	int					in_nValue,
	double				in_prePeakFinderSliceK,
	int					in_thresholdValueOfMaxSignalValue,
	int					in_amountOfParams,
	int					in_discreteModelMode,
	int					in_prescaller,
	double* in_params,
	int					in_mode,
	double				in_memoryLimit,
	std::string			in_outPath,
	bool				in_debug,
	std::atomic<int>& progress);



// Обертка, которая для каждой бифуркационки будет своя!
__host__ void bifurcation2D(double					in_tMax,
	int					in_nPts,
	double				in_h1,
	double				in_h2,
	double* in_initialConditions,
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
	double* in_params,
	int					in_mode1,
	int					in_mode2,
	int					in_kdeSampling,
	float				in_kdeSamplesInterval1,
	float				in_kdeSamplesInterval2,
	float				in_kdeSamplesSmooth,
	double				in_memoryLimit,
	std::string			in_outPath,
	bool				in_debug,
	std::atomic<int>& progress);



// Обертка, которая для каждой бифуркационки будет своя!
__host__ void bifurcation3D(
	double					in_tMax,
	int					in_nPts,
	double				in_h,
	double* in_initialConditions,
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
	double* in_params,
	int					in_mode1,
	int					in_mode2,
	int					in_mode3,
	int					in_kdeSampling,
	float				in_kdeSamplesInterval1,
	float				in_kdeSamplesInterval2,
	float				in_kdeSamplesSmooth,
	double				in_memoryLimit,
	std::string			in_outPath,
	bool				in_debug,
	std::atomic<int>& progress);



__device__ int peakFinder(int idx,
	float prePeakFinder,
	size_t amountOfTPoints,
	float* in_data,
	int* out_dataSizes,
	float* out_data);

__device__ int peakFinderForDBSCAN(int idx,
	float in_h,
	float prePeakFinder,
	size_t amountOfTPoints,
	float* in_data,
	float* out_data,
	int* out_dataSizes);


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


__device__ float distance(float x1, float y1, float x2, float y2);

__device__ void expand_cluster(float* input, int index, int amountOfPeaks, int p, float eps);

__device__ int dbscan(float* input, int amountOfTPoints, int amountOfPeaks, int idx, float eps, int* dataSizes, int criticalValueOfPeaks);

__device__ float customAbs(float value);

// return in "out" array [a, b] with "amount" elements
template <class T1, class T2>
__host__ void linspace(T1 a, T1 b, int amount, T2* out, int startIndex = 0, bool isExp = false);

__host__ void getParamsAndSymmetry2D(double* param1, double* param2,
	double startInterval1, double finishInteraval1,
	double startInterval2, double finishInteraval2,
	int nPts, bool isExpForParam1 = false);

__host__ void getParamsAndSymmetry3D(double* param1, double* param2, double* param3,
	double startInterval1, double finishInteraval1,
	double startInterval2, double finishInteraval2,
	double startInterval3, double finishInteraval3,
	int nPts, bool isExpForParam1=false);

template <class T>
__host__ void slice(T* in, int a, int b, T* out);