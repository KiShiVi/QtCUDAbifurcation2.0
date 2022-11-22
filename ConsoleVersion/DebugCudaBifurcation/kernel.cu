#include "C:\GitHub\QtCUDAbifurcation2.0\Library\bifurcationKernel.cuh"
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
	//std::string str = "C:\\CUDA\\My\\mat_h_";
	//str = str + std::to_string((float)0.001) + ".csv";
	std::atomic<int> a;
	a.store(0);
	double initialConditions[4]{ -0.2f,-0.2f,-0.2f,0.2f };
	////double params[3]{ 0.5,1.0,1.0 };
	////double params[4]{ 0.5,10,28,2.7 };
	double params[4]{ 0.5,0.2,0.9,5.7 };
	int AmountPar = sizeof(params) / sizeof(double);
	//

	//	bifurcation1D(
	//	200,		//int					in_tMax,
	//	500,		//int					in_nPts,
	//	1e-2,		//double				in_h,
	//	initialConditions,		//double* in_initialConditions,
	//	0.05,		//double				in_paramValues1,
	//	0.45,		//double				in_paramValues2,
	//	0,		//int					in_nValue,
	//	4500,		//double				in_prePeakFinderSliceK,
	//	200000,		//int					in_thresholdValueOfMaxSignalValue,
	//	AmountPar,		//int					in_amountOfParams,
	//	DISSIPATA,		//int					in_discreteModelMode,
	//	1,		//int					in_prescaller,
	//	params,		//double* in_params,
	//	1,		//int					in_mode,
	//	0.9,		//double				in_memoryLimit,
	//	"C:\\CUDA\\My\\mat1d.csv",		//std::string			in_outPath,
	//	1,		//bool				in_debug,
	//	a		//std::atomic<int> & progress);
	//);

			bifurcation2D(
			200,		//int					in_tMax,
			200,		//int					in_nPts,
			0.01,		//double				in_h,
			initialConditions,		//double* in_initialConditions,
			0.05,		//double				in_paramValues1,
			0.45,		//double				in_paramValues2,
			0.05,			//double				in_paramValues3,
			1,			//double				in_paramValues4,
			0,			//int					in_nValue,
			1000,		//double				in_prePeakFinderSliceK, 
			200000,		//int					in_thresholdValueOfMaxSignalValue,
			AmountPar,	//int					in_amountOfParams,
			DISSIPATA,		//int					in_discreteModelMode,
			1,			//int					in_prescaller,
			params,		//double* in_params,
			1,			//int					in_mode1,
			2,			//int					in_mode2,
			10,			//int					in_kdeSampling,
			-20,		//double				in_kdeSamplesInterval1,
			70,			//double				in_kdeSamplesInterval2,
			5e-2,		//double				in_kdeSamplesSmooth,
			0.95,		//double				in_memoryLimit,
			"C:\\CUDA\\My\\mat_h_2D.csv", //std::string			in_outPath,
			1,		//bool				in_debug,
			a		//std::atomic<int> & progress);
		);


	////bifurcation1D(
	////	300,		//int					in_tMax,
	////	200,		//int					in_nPts,
	////	5e-3,		//double				in_h,
	////	initialConditions,		//double* in_initialConditions,
	////	0,		//double				in_paramValues1,
	////	4,		//double				in_paramValues2,
	////	0,		//int					in_nValue,
	////	500,		//double				in_prePeakFinderSliceK,
	////	200000,		//int					in_thresholdValueOfMaxSignalValue,
	////	AmountPar,		//int					in_amountOfParams,
	////	DISSIPATA,		//int					in_discreteModelMode,
	////	1,		//int					in_prescaller,
	////	params,		//double* in_params,
	////	3,		//int					in_mode,
	////	0.9,		//double				in_memoryLimit,
	////	"C:\\CUDA\\My\\mat1d.csv",		//std::string			in_outPath,
	////	1,		//bool				in_debug,
	////	a		//std::atomic<int> & progress);
	////);
	//double values_h[5]{ 0.05, 0.1,0.15,0.2,0.25};
	////int values_prescaller[9]{5,3,2,2,1,1,1,1,1};
	////double values_h[34]{3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05 };
	////int values_prescaller[34]{1666,1250,1000,833,714,625,555,500,250,166,125,100,83,71,62,55,50,25,16,12,10,8,7,6,5,5,3,2,2,1,1,1,1,1 };
	//std::string str = "C:\\CUDA\\My\\CD_mat_h_";
	//std::string path;
	//for (int i = 0; i < 5; i++) {
	//	path = str + std::to_string(i) + ".csv";
	//	size_t startTime = clock();
	//	bifurcation2D(
	//		300,		//int					in_tMax,
	//		400,		//int					in_nPts,
	//		values_h[i],		//double				in_h,
	//		initialConditions,		//double* in_initialConditions,
	//		0,		//double				in_paramValues1,
	//		0.45,		//double				in_paramValues2,
	//		0,			//double				in_paramValues3,
	//		1,			//double				in_paramValues4,
	//		0,			//int					in_nValue,
	//		5000,		//double				in_prePeakFinderSliceK, 
	//		200000,		//int					in_thresholdValueOfMaxSignalValue,
	//		AmountPar,	//int					in_amountOfParams,
	//		DISSIPATA,		//int					in_discreteModelMode,
	//		1,			//int					in_prescaller,
	//		params,		//double* in_params,
	//		1,			//int					in_mode1,
	//		2,			//int					in_mode2,
	//		10,			//int					in_kdeSampling,
	//		-20,		//double				in_kdeSamplesInterval1,
	//		70,			//double				in_kdeSamplesInterval2,
	//		5e-2,		//double				in_kdeSamplesSmooth,
	//		1,		//double				in_memoryLimit,
	//		path, //std::string			in_outPath,
	//		1,		//bool				in_debug,
	//		a		//std::atomic<int> & progress);
	//	);

	//	//bifurcation2D(
	//	//	500,		//int					in_tMax,
	//	//	100,		//int					in_nPts,
	//	//	values_h[i],		//double				in_h,
	//	//	initialConditions,		//double* in_initialConditions,
	//	//	0,		//double				in_paramValues1,
	//	//	2,		//double				in_paramValues2,
	//	//	0,			//double				in_paramValues3,
	//	//	2,			//double				in_paramValues4,
	//	//	0,			//int					in_nValue,
	//	//	100000,		//double				in_prePeakFinderSliceK, 
	//	//	200000,		//int					in_thresholdValueOfMaxSignalValue,
	//	//	AmountPar,	//int					in_amountOfParams,
	//	//	CONSERVA,		//int					in_discreteModelMode,
	//	//	1,			//int					in_prescaller,
	//	//	params,		//double* in_params,
	//	//	1,			//int					in_mode1,
	//	//	2,			//int					in_mode2,
	//	//	10,			//int					in_kdeSampling,
	//	//	-5,		//double				in_kdeSamplesInterval1,
	//	//	15,			//double				in_kdeSamplesInterval2,
	//	//	1e-2,		//double				in_kdeSamplesSmooth,
	//	//	1,		//double				in_memoryLimit,
	//	//	path, //std::string			in_outPath,
	//	//	1,		//bool				in_debug,
	//	//	a		//std::atomic<int> & progress);
	//	//);

	//	std::cout << clock() - startTime << " ms\n";
	//}









	//double L = 430e-6;
//	double dL = 1;
//	double Lx = L;
//	double RL = 2.8;
//	double om = 12e3;
//	double C2 = 10e-9;
//	double R2 = 1e3;
//	double ay = 100;
//	double mu = 100;
//	double ax = 100;
//	double az = ax / om / Lx;
//	double R3 = 10 * L * om * ax * az * R2 / ay / mu;
//	double R6 = mu * az / C2 / ax / om;
//	double R7 = mu * az / ay / C2 / om;
//	double R1 = 2.3;
//	double R8 = -4.8;
//	double params[12] = { 0.5,R8,RL,1/L,1/dL,mu,R3,1 / R2,R1,1 / R6,1 / R7,1 / C2 };
//	double initialConditions[4]{ 0.001f, 0.0f, 0.0f, 0.0f};
//	int AmountPar = sizeof(params) / sizeof(double);
//
////	bifurcation1D(
////	0.1,		//int					in_tMax,
////	1000,		//int					in_nPts,
////	1e-5,		//double				in_h,
////	initialConditions,		//double* in_initialConditions,
////	0,		//double				in_paramValues1,
////	10,		//double				in_paramValues2,
////	2,		//int					in_nValue,
////	1,		//double				in_prePeakFinderSliceK,
////	200000,		//int					in_thresholdValueOfMaxSignalValue,
////	AmountPar,		//int					in_amountOfParams,
////	TIMUR,		//int					in_discreteModelMode,
////	1,		//int					in_prescaller,
////	params,		//double* in_params,
////	8,		//int					in_mode,
////	0.9,		//double				in_memoryLimit,
////	"C:\\CUDA\\My\\mat_1D.csv",		//std::string			in_outPath,
////	1,		//bool				in_debug,
////	a		//std::atomic<int> & progress);
////);
//
//	bifurcation2D(
//		0.1,		//int					in_tMax,
//		100,		//int					in_nPts,
//		1e-5,		//double				in_h,
//		initialConditions,		//double* in_initialConditions,
//		1400,		//double				in_paramValues1,
//		3600,			//double				in_paramValues2,
//		0,			//double				in_paramValues3,
//		8,			//double				in_paramValues4,
//		2,			//int					in_nValue,
//		1,		//double				in_prePeakFinderSliceK, 
//		200000,		//int					in_thresholdValueOfMaxSignalValue,
//		AmountPar,	//int					in_amountOfParams,
//		TIMUR,		//int					in_discreteModelMode,
//		1,			//int					in_prescaller,
//		params,		//double* in_params,
//		3,			//int					in_mode1,
//		8,			//int					in_mode2,
//		20,			//int					in_kdeSampling,
//		-2,		//double				in_kdeSamplesInterval1,
//		2,			//double				in_kdeSamplesInterval2,
//		2e-4,		//double				in_kdeSamplesSmooth,
//		0.8,		//double				in_memoryLimit,
//		"C:\\CUDA\\My\\mat_2D.csv", //std::string			in_outPath,
//		1,		//bool				in_debug,
//		a		//std::atomic<int> & progress);
//	);
}
