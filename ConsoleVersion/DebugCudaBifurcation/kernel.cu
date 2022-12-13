#include "..\..\Library\bifurcationKernel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <atomic>
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>

int main()
{

	std::atomic<int> a;
	a.store(0);
	double initialConditions[3]{ 0.1, 0.1, 30 };

	double params[5]{ 2, 0.5, 40, 3, 28 };

	int AmountPar = sizeof(params) / sizeof(double);
	//
	size_t startTime = clock();


	bifurcation2D(
		100,					//int					in_tMax,
		500,					//int					in_nPts,
		3,					//double				in_h1,
		1.3979,					//double				in_h2,
		initialConditions,		//double* in_initialConditions,
		-0.5,						//double				in_paramValues1,
		1,						//double				in_paramValues2,
		-0.006,						//double				in_paramValues3,
		0.004,						//double				in_paramValues4,
		0,						//int					in_nValue,
		500,					//double				in_prePeakFinderSliceK, 
		200000,					//int					in_thresholdValueOfMaxSignalValue,
		AmountPar,				//int					in_amountOfParams,
		CHEN,				//int					in_discreteModelMode,
		1,						//int					in_prescaller,
		params,					//double* in_params,
		0,						//int					in_mode1,
		1,						//int					in_mode2,
		10,						//int					in_kdeSampling,
		-5,						//double				in_kdeSamplesInterval1,
		15,						//double				in_kdeSamplesInterval2,
		1e-2,					//double				in_kdeSamplesSmooth,
		0.95,					//double				in_memoryLimit,
		"C:\\Users\\KiShiVi\\Desktop\\mat.csv", //std::string			in_outPath,
		1,						//bool				in_debug,
		a						//std::atomic<int> & progress);
	);



	std::cout << clock() - startTime << " ms\n";
}
