#include "C:\GitHub\QtCUDAbifurcation2.0\Library\bifurcationKernel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <atomic>
#include <ctime>
#include <iostream>
#include <fstream>

int main()
{
	std::string str = "STRING";
	str = str + (std::string)2;
	std::atomic<int> a;
	a.store(0);
	double initialConditions[4]{ 0.2f,0.2f,0.2f,0.2f };
	double params[3]{ 0.5f,1.0f,1.0f };
	int AmountPar = sizeof(params) / sizeof(double);
	size_t startTime = clock();

	//bifurcation1D(
	//	1000,		//int					in_tMax,
	//	1000,		//int					in_nPts,
	//	5e-2,		//double				in_h,
	//	initialConditions,		//double* in_initialConditions,
	//	0,		//double				in_paramValues1,
	//	2,		//double				in_paramValues2,
	//	0,		//int					in_nValue,
	//	20000,		//double				in_prePeakFinderSliceK,
	//	200000,		//int					in_thresholdValueOfMaxSignalValue,
	//	AmountPar,		//int					in_amountOfParams,
	//	DEFF,		//int					in_discreteModelMode,
	//	1,		//int					in_prescaller,
	//	params,		//double* in_params,
	//	1,		//int					in_mode,
	//	0.9,		//double				in_memoryLimit,
	//	"C:\\CUDA\\My\\mat1d.csv",		//std::string			in_outPath,
	//	1,		//bool				in_debug,
	//	a		//std::atomic<int> & progress);
	//);

	bifurcation2D(
		500,		//int					in_tMax,
		80,		//int					in_nPts,
		5e-5,		//double				in_h,
		initialConditions,		//double* in_initialConditions,
		0,		//double				in_paramValues1,
		2,		//double				in_paramValues2,
		0,			//double				in_paramValues3,
		2,			//double				in_paramValues4,
		0,			//int					in_nValue,
		1000,		//double				in_prePeakFinderSliceK, 
		200000,		//int					in_thresholdValueOfMaxSignalValue,
		AmountPar,	//int					in_amountOfParams,
		DEFF,		//int					in_discreteModelMode,
		1000,			//int					in_prescaller,
		params,		//double* in_params,
		1,			//int					in_mode1,
		2,			//int					in_mode2,
		10,			//int					in_kdeSampling,
		-5,		//double				in_kdeSamplesInterval1,
		15,			//double				in_kdeSamplesInterval2,
		1e-2,		//double				in_kdeSamplesSmooth,
		1,		//double				in_memoryLimit,
		"C:\\CUDA\\My\\mat.csv", //std::string			in_outPath,
		1,		//bool				in_debug,
		a		//std::atomic<int> & progress);
	);

	std::cout << clock() - startTime << " ms\n";
}
