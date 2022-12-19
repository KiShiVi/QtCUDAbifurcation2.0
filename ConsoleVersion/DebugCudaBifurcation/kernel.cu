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

	std::atomic<int> a;
	a.store(0);
	size_t startTime = clock();
	double initialConditions[3]{ 0.001, 0, 0 };
	double params[4]{ 0.5, 0.2, 0.2, 5.7 };
	int AmountPar = sizeof(params) / sizeof(double);


	double	in_tMax = 500;
	double  in_NT = 0.4;
	int		in_nPts = 2000;
	double	in_h = 0.01;
	double	in_paramValues1 = 0.05;
	double	in_paramValues2 = 0.38;
	double	in_paramValues3 = 0;
	double	in_paramValues4 = 20;
	int		in_nValue = 1;
	double	in_prePeakFinderSliceK = 4000;
	int		in_thresholdValueOfMaxSignalValue = 200000;
	int		in_prescaller = 1;
	int		in_mode1 = 1;
	int		in_mode2 = 2;
	double	in_memoryLimit = 0.5;
	double eps = 1e-8;


	//bifurcation1D(
	//	in_tMax,
	//	in_nPts,
	//	in_h,
	//	initialConditions,
	//	in_paramValues1,
	//	in_paramValues2,
	//	in_nValue,
	//	in_prePeakFinderSliceK,
	//	in_thresholdValueOfMaxSignalValue,
	//	AmountPar,
	//	ROSSLER,
	//	in_prescaller,
	//	params,
	//	in_mode1,
	//	in_memoryLimit,
	//	"C:\\CUDA\\My\\bif.csv",
	//	1,
	//	a
	//);

			LLE1D(
			in_tMax,
			in_NT,
			in_nPts,
			in_h,
			initialConditions,
			in_paramValues1,
			in_paramValues2,
			in_prePeakFinderSliceK,
			in_thresholdValueOfMaxSignalValue,
			AmountPar,
			ROSSLER,
			in_prescaller,
			eps,
			params,
			in_mode1,
			in_memoryLimit,
			"C:\\CUDA\\My\\LLE.csv",
			1,
			a);

		//double initialConditions[6]{ 0.001, 0, 0, 0.02, 0.02, 0.02 };
		//double params[11]{ 0.5, -1, 1, 0.5, 7, 0, 0.5, 0.143, 0.2, 5.7, 4.09};
		//int AmountPar = sizeof(params) / sizeof(double);

		//double	in_tMax = 500;
		//int		in_nPts = 200;
		//double	in_h = 0.01;
		//double	in_paramValues1 = 0.1;
		//double	in_paramValues2 = 0.17;
		//double	in_paramValues3 = 0;
		//double	in_paramValues4 = 20;
		//int		in_nValue = 1;
		//double	in_prePeakFinderSliceK = 4000;
		//int		in_thresholdValueOfMaxSignalValue = 200000;
		//int		in_prescaller = 1;
		//int		in_mode1 = 7;
		//int		in_mode2 = 4;
		//double	in_memoryLimit = 0.95;



		//bifurcation1D(
		//	in_tMax,
		//	20000,
		//	in_h,
		//	initialConditions,
		//	in_paramValues1,
		//	in_paramValues2,
		//	in_nValue,
		//	in_prePeakFinderSliceK,
		//	in_thresholdValueOfMaxSignalValue,
		//	AmountPar,
		//	DUFFING_RO,
		//	in_prescaller,
		//	params,
		//	in_mode1,
		//	in_memoryLimit,
		//	"C:\\CUDA\\My\\bif_1d_duffro_1.csv",
		//	1,
		//	a
		//);


		//bifurcation2D(
		//	in_tMax,
		//	in_nPts,
		//	in_h,
		//	initialConditions,
		//	in_paramValues1,
		//	in_paramValues2,
		//	in_paramValues3,
		//	in_paramValues4,
		//	in_nValue,
		//	in_prePeakFinderSliceK,
		//	in_thresholdValueOfMaxSignalValue,
		//	AmountPar,
		//	DUFFING_RO,
		//	in_prescaller,
		//	params,
		//	in_mode1,
		//	in_mode2,
		//	1,
		//	1,
		//	1,
		//	1,
		//	in_memoryLimit,
		//	"C:\\CUDA\\My\\bif_1d_duffro_2d.csv",
		//	1,
		//	a);

		//std::atomic<int> a;
		//a.store(0);
		//double initialConditions[4]{ 0.01, 0.01, 0.01, 0.01 };

		//double params[7]{ 0.5, 6, 10, 2.7, 1, 1.5, 1 };

		//int AmountPar = sizeof(params) / sizeof(double);
		////
		//size_t startTime = clock();
		//bifurcation2D(
		//	500, //int in_tMax,
		//	150, //int in_nPts,
		//	0.01, //double in_h,
		//	initialConditions, //double* in_initialConditions,
		//	4, //double in_paramValues1,
		//	25, //double in_paramValues2,
		//	6, //double in_paramValues3,
		//	14, //double in_paramValues4,
		//	0, //int in_nValue,
		//	1000, //double in_prePeakFinderSliceK,
		//	200000, //int in_thresholdValueOfMaxSignalValue,
		//	AmountPar, //int in_amountOfParams,
		//	CD, //int in_discreteModelMode,
		//	1, //int in_prescaller,
		//	params, //double* in_params,
		//	1, //int in_mode1,
		//	2, //int in_mode2,
		//	10, //int in_kdeSampling,
		//	-5, //double in_kdeSamplesInterval1,
		//	15, //double in_kdeSamplesInterval2,
		//	1e-2, //double in_kdeSamplesSmooth,
		//	0.95, //double in_memoryLimit,
		//	"C:\\CUDA\\My\\bif_2d.csv", //std::string in_outPath,
		//	1, //bool in_debug,
		//	a //std::atomic<int> & progress);
		//);



		//double values_h[5]{ 0.005,0.0075,0.01,0.0125,0.015 };
		////int values_prescaller[5]{ 1,1,1,1,1 };
		////double values_h[34]{3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05 };
		////int values_prescaller[34]{1666,1250,1000,833,714,625,555,500,250,166,125,100,83,71,62,55,50,25,16,12,10,8,7,6,5,5,3,2,2,1,1,1,1,1 };
		//std::string str = "C:\\CUDA\\My\\RK4_mat_h_";
		//std::string path;

		//int CASE = 8;
		//for (int i = 0; i < 2; i++) {
		//	CASE++;
		//	if (CASE == 9)
		//		str = "C:\\CUDA\\My\\RK2_mat_h_";
		//	if (CASE == 10)
		//		str = "C:\\CUDA\\My\\CD_mat_h_";
		//	for (int i = 0; i < 5; i++) {


		//		path = str + std::to_string(i) + ".csv";
		//		size_t startTime = clock();
		//		bifurcation2D(
		//			400, //int in_tMax,
		//			200, //int in_nPts,
		//			values_h[i], //double in_h,
		//			initialConditions, //double* in_initialConditions,
		//			4, //double in_paramValues1,
		//			25, //double in_paramValues2,
		//			6, //double in_paramValues3,
		//			14, //double in_paramValues4,
		//			0, //int in_nValue,
		//			1000, //double in_prePeakFinderSliceK,
		//			200000, //int in_thresholdValueOfMaxSignalValue,
		//			AmountPar, //int in_amountOfParams,
		//			CASE, //int in_discreteModelMode,
		//			1, //int in_prescaller,
		//			params, //double* in_params,
		//			1, //int in_mode1,
		//			2, //int in_mode2,
		//			10, //int in_kdeSampling,
		//			-5, //double in_kdeSamplesInterval1,
		//			15, //double in_kdeSamplesInterval2,
		//			1e-2, //double in_kdeSamplesSmooth,
		//			0.95, //double in_memoryLimit,
		//			path, //std::string in_outPath,
		//			1, //bool in_debug,
		//			a //std::atomic<int> & progress);
		//		);
		//		std::cout << clock() - startTime << " ms\n";
		//	}
		//}


		std::cout << clock() - startTime << " ms\n";
}




//int main()
//{
//	//std::string str = "C:\\CUDA\\My\\mat_h_";
//	//str = str + std::to_string((float)0.001) + ".csv";
//	std::atomic<int> a;
//	a.store(0);
//	double initialConditions[4]{ -0.2,-0.2,-0.2,0.2 };
//	//double initialConditions[3]{ 0.001,0,0 };
//	//double params[3]{ 0.5,1.0,1.0 };
//	double params[4]{ 0.5,0.2,0.2,5.7 };
//	//double params[4]{ 0.5,10,28,2.667 };
//	//double params[4]{ 0.5,0.2,0.2,5.7 };
//	//double params[6]{ 2.7,0.75,1.2,6.9,0.367,0.0478 };
//	int AmountPar = sizeof(params) / sizeof(double);
//	//
//	size_t startTime = clock();
//	//	bifurcation1D(
//	//	200,		//int					in_tMax,
//	//	500,		//int					in_nPts,
//	//	1e-2,		//double				in_h,
//	//	initialConditions,		//double* in_initialConditions,
//	//	2,		//double				in_paramValues1,
//	//	3,		//double				in_paramValues2,
//	//	0,		//int					in_nValue,
//	//	50,		//double				in_prePeakFinderSliceK,
//	//	2000000,		//int					in_thresholdValueOfMaxSignalValue,
//	//	AmountPar,		//int					in_amountOfParams,
//	//	DISSIPATA,		//int					in_discreteModelMode,
//	//	1,		//int					in_prescaller,
//	//	params,		//double* in_params,
//	//	0,		//int					in_mode,
//	//	0.9,		//double				in_memoryLimit,
//	//	"C:\\CUDA\\My\\mat1d.csv",		//std::string			in_outPath,
//	//	1,		//bool				in_debug,
//	//	a		//std::atomic<int> & progress);
//	//);
//
//	//	bifurcation3D(
//	//	300, //double					in_tMax,
//	//	100, //int					in_nPts,
//	//	1e-2, //double				in_h,
//	//	initialConditions, //double* in_initialConditions,
//	//	2,//				in_paramValues1,
//	//	3,//				in_paramValues2,
//	//	0.5,//				in_paramValues3,
//	//	1.0,//				in_paramValues4,
//	//	1,//double				in_paramValues5,
//	//	2,//double				in_paramValues6,
//	//	1,//int					in_nValue,
//	//	50,//double				in_prePeakFinderSliceK,
//	//	20000, //int					in_thresholdValueOfMaxSignalValue,
//	//	AmountPar, //int					in_amountOfParams,
//	//	DISSIPATA, //int					in_discreteModelMode,
//	//	1,//int					in_prescaller,
//	//	params, //double* in_params,
//	//	0, //int					in_mode1,
//	//	1, //int					in_mode2,
//	//	2, //int					in_mode3,
//	//	10, //int					in_kdeSampling,
//	//	-10, //float				in_kdeSamplesInterval1,
//	//	20, //float				in_kdeSamplesInterval2,
//	//	1e-2, //float				in_kdeSamplesSmooth,
//	//	0.95, //double				in_memoryLimit,
//	//	"C:\\CUDA\\My\\mat_h_3D.csv", //std::string			in_outPath,
//	//	1, //bool				in_debug,
//	//	a //std::atomic<int> & progress
//	//);
//
//			//bifurcation2D(
//			//	1000,					//int					in_tMax,
//			//	2000,					//int					in_nPts,
//			//	0.1,					//double				in_h,
//			//	initialConditions,		//double* in_initialConditions,
//			//	0,						//double				in_paramValues1,
//			//	4,						//double				in_paramValues2,
//			//	-2,						//double				in_paramValues3,
//			//	6,						//double				in_paramValues4,
//			//	0,						//int					in_nValue,
//			//	1000,					//double				in_prePeakFinderSliceK, 
//			//	200000,					//int					in_thresholdValueOfMaxSignalValue,
//			//	AmountPar,				//int					in_amountOfParams,
//			//	7,				//int					in_discreteModelMode,
//			//	1,						//int					in_prescaller,
//			//	params,					//double* in_params,
//			//	1,						//int					in_mode1,
//			//	2,						//int					in_mode2,
//			//	10,						//int					in_kdeSampling,
//			//	-5,						//double				in_kdeSamplesInterval1,
//			//	15,						//double				in_kdeSamplesInterval2,
//			//	1e-2,					//double				in_kdeSamplesSmooth,
//			//	0.95,					//double				in_memoryLimit,
//			//	"C:\\CUDA\\My\\mat_h_2D.csv", //std::string			in_outPath,
//			//	1,						//bool				in_debug,
//			//	a						//std::atomic<int> & progress);
//			//);
//
//
//
//	double values_h[5]{ 0.1, 0.125,0.15,0.175,0.2 };
//	int values_prescaller[5]{ 1,1,1,1,1 };
//	//double values_h[34]{3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05 };
//	//int values_prescaller[34]{1666,1250,1000,833,714,625,555,500,250,166,125,100,83,71,62,55,50,25,16,12,10,8,7,6,5,5,3,2,2,1,1,1,1,1 };
//	std::string str = "C:\\CUDA\\My\\RK4_mat_h_";
//	std::string path;
//
//			bifurcation2D(
//				300,					//int					in_tMax,
//				5000,					//int					in_nPts,
//				0.05,					//double				in_h,
//				initialConditions,		//double* in_initialConditions,
//				0.1,						//double				in_paramValues1,
//				0.4,						//double				in_paramValues2,
//				0,						//double				in_paramValues3,
//				0.7,						//double				in_paramValues4,
//				0,						//int					in_nValue,
//				500,					//double				in_prePeakFinderSliceK, 
//				200000,					//int					in_thresholdValueOfMaxSignalValue,
//				AmountPar,				//int					in_amountOfParams,
//				5,				//int					in_discreteModelMode,
//				1,						//int					in_prescaller,
//				params,					//double* in_params,
//				1,						//int					in_mode1,
//				2,						//int					in_mode2,
//				10,						//int					in_kdeSampling,
//				-5,						//double				in_kdeSamplesInterval1,
//				15,						//double				in_kdeSamplesInterval2,
//				1e-2,					//double				in_kdeSamplesSmooth,
//				0.95,					//double				in_memoryLimit,
//				"C:\\CUDA\\My\\mat_ComCD_2D.csv", //std::string			in_outPath,
//				1,						//bool				in_debug,
//				a						//std::atomic<int> & progress);
//			);
//
//
//	//int CASE = 6;
//	//for (int i = 0; i < 2; i++) {
//	//	CASE++;
//	//	if (CASE == 7)
//	//		str = "C:\\CUDA\\My\\CompCD_mat_h_";
//	//	if (CASE == 8)
//	//		str = "C:\\CUDA\\My\\RK4_mat_h_";
//	//	for (int i = 0; i < 5; i++) {
//
//
//	//		path = str + std::to_string(i) + ".csv";
//	//		size_t startTime = clock();
//	//		bifurcation2D(
//	//			1000,					//int					in_tMax,
//	//			400,					//int					in_nPts,
//	//			values_h[i],					//double				in_h,
//	//			initialConditions,		//double* in_initialConditions,
//	//			0,						//double				in_paramValues1,
//	//			2,						//double				in_paramValues2,
//	//			0,						//double				in_paramValues3,
//	//			2,						//double				in_paramValues4,
//	//			0,						//int					in_nValue,
//	//			2000,					//double				in_prePeakFinderSliceK, 
//	//			200000,					//int					in_thresholdValueOfMaxSignalValue,
//	//			AmountPar,				//int					in_amountOfParams,
//	//			CASE,				//int					in_discreteModelMode,
//	//			values_prescaller[i],						//int					in_prescaller,
//	//			params,					//double* in_params,
//	//			1,						//int					in_mode1,
//	//			2,						//int					in_mode2,
//	//			10,						//int					in_kdeSampling,
//	//			-5,						//double				in_kdeSamplesInterval1,
//	//			15,						//double				in_kdeSamplesInterval2,
//	//			1e-2,					//double				in_kdeSamplesSmooth,
//	//			0.95,					//double				in_memoryLimit,
//	//			path, //std::string			in_outPath,
//	//			1,						//bool				in_debug,
//	//			a						//std::atomic<int> & progress);
//	//		);
//
//	//		//bifurcation2D(
//	//		//	500,		//int					in_tMax,
//	//		//	100,		//int					in_nPts,
//	//		//	values_h[i],		//double				in_h,
//	//		//	initialConditions,		//double* in_initialConditions,
//	//		//	0,		//double				in_paramValues1,
//	//		//	2,		//double				in_paramValues2,
//	//		//	0,			//double				in_paramValues3,
//	//		//	2,			//double				in_paramValues4,
//	//		//	0,			//int					in_nValue,
//	//		//	100000,		//double				in_prePeakFinderSliceK, 
//	//		//	200000,		//int					in_thresholdValueOfMaxSignalValue,
//	//		//	AmountPar,	//int					in_amountOfParams,
//	//		//	CONSERVA,		//int					in_discreteModelMode,
//	//		//	1,			//int					in_prescaller,
//	//		//	params,		//double* in_params,
//	//		//	1,			//int					in_mode1,
//	//		//	2,			//int					in_mode2,
//	//		//	10,			//int					in_kdeSampling,
//	//		//	-5,		//double				in_kdeSamplesInterval1,
//	//		//	15,			//double				in_kdeSamplesInterval2,
//	//		//	1e-2,		//double				in_kdeSamplesSmooth,
//	//		//	1,		//double				in_memoryLimit,
//	//		//	path, //std::string			in_outPath,
//	//		//	1,		//bool				in_debug,
//	//		//	a		//std::atomic<int> & progress);
//	//		//);
//
//	//		std::cout << clock() - startTime << " ms\n";
//	//	}
//	//}
//
//
//
//
//
//
//
//
//
//
//	//double L = 430e-6;
////	double dL = 1;
////	double Lx = L;
////	double RL = 2.8;
////	double om = 12e3;
////	double C2 = 10e-9;
////	double R2 = 1e3;
////	double ay = 100;
////	double mu = 100;
////	double ax = 100;
////	double az = ax / om / Lx;
////	double R3 = 10 * L * om * ax * az * R2 / ay / mu;
////	double R6 = mu * az / C2 / ax / om;
////	double R7 = mu * az / ay / C2 / om;
////	double R1 = 2.3;
////	double R8 = -4.8;
////	double params[12] = { 0.5,R8,RL,1/L,1/dL,mu,R3,1 / R2,R1,1 / R6,1 / R7,1 / C2 };
////	double initialConditions[4]{ 0.001f, 0.0f, 0.0f, 0.0f};
////	int AmountPar = sizeof(params) / sizeof(double);
////
//////	bifurcation1D(
//////	0.1,		//int					in_tMax,
//////	1000,		//int					in_nPts,
//////	1e-5,		//double				in_h,
//////	initialConditions,		//double* in_initialConditions,
//////	0,		//double				in_paramValues1,
//////	10,		//double				in_paramValues2,
//////	2,		//int					in_nValue,
//////	1,		//double				in_prePeakFinderSliceK,
//////	200000,		//int					in_thresholdValueOfMaxSignalValue,
//////	AmountPar,		//int					in_amountOfParams,
//////	TIMUR,		//int					in_discreteModelMode,
//////	1,		//int					in_prescaller,
//////	params,		//double* in_params,
//////	8,		//int					in_mode,
//////	0.9,		//double				in_memoryLimit,
//////	"C:\\CUDA\\My\\mat_1D.csv",		//std::string			in_outPath,
//////	1,		//bool				in_debug,
//////	a		//std::atomic<int> & progress);
//////);
////
////	bifurcation2D(
////		0.1,		//int					in_tMax,
////		100,		//int					in_nPts,
////		1e-5,		//double				in_h,
////		initialConditions,		//double* in_initialConditions,
////		1400,		//double				in_paramValues1,
////		3600,			//double				in_paramValues2,
////		0,			//double				in_paramValues3,
////		8,			//double				in_paramValues4,
////		2,			//int					in_nValue,
////		1,		//double				in_prePeakFinderSliceK, 
////		200000,		//int					in_thresholdValueOfMaxSignalValue,
////		AmountPar,	//int					in_amountOfParams,
////		TIMUR,		//int					in_discreteModelMode,
////		1,			//int					in_prescaller,
////		params,		//double* in_params,
////		3,			//int					in_mode1,
////		8,			//int					in_mode2,
////		20,			//int					in_kdeSampling,
////		-2,		//double				in_kdeSamplesInterval1,
////		2,			//double				in_kdeSamplesInterval2,
////		2e-4,		//double				in_kdeSamplesSmooth,
////		0.8,		//double				in_memoryLimit,
////		"C:\\CUDA\\My\\mat_2D.csv", //std::string			in_outPath,
////		1,		//bool				in_debug,
////		a		//std::atomic<int> & progress);
////	);
//std::cout << clock() - startTime << " ms\n";
//}
