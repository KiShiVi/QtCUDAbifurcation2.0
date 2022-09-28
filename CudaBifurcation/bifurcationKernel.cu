﻿#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "bifurcationKernel.cuh"

__host__ void bifurcation1D(	int					in_tMax,
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
								std::atomic<int>&	progress)
{
	size_t amountOfTPoints = in_tMax / in_h;

	float* globalParamValues = nullptr;
	globalParamValues = (float*)malloc(sizeof(float) * in_nPts);
	linspace(in_paramValues1, in_paramValues2, in_nPts, globalParamValues);

	float* aParamValues			= nullptr;
	float* bParamValues			= nullptr;
	float* cParamValues			= nullptr;
	float* symmetryParamValues	= nullptr;

	aParamValues		= (float*)malloc(sizeof(float) * in_nPts);
	bParamValues		= (float*)malloc(sizeof(float) * in_nPts);
	cParamValues		= (float*)malloc(sizeof(float) * in_nPts);
	symmetryParamValues = (float*)malloc(sizeof(float) * in_nPts);

	getParamsAndSymmetry(aParamValues, bParamValues, cParamValues, symmetryParamValues,
						in_paramA, in_paramB, in_paramC, 0.5f,
						in_paramValues1, in_paramValues2, in_mode, in_nPts);

	size_t freeMemory;
	size_t totalMemory;

	cudaMemGetInfo(&freeMemory, &totalMemory);

	freeMemory *= in_memoryLimit;

	float maxMemoryLimit = sizeof(float) * ((in_tMax / in_h) + 4) + sizeof(int);

	size_t nPtsLimiter = freeMemory / maxMemoryLimit;

	if (nPtsLimiter <= 0)
	{
		if (in_debug)
			std::cout << "\nVery low memory size. Increase the MEMORY_LIMIT!" << "\n";
		exit(1);
	}

	float	*	h_data;
	int		*	h_dataSizes;
	float	*	h_dataTimes;
	float	*	h_a_dataTimes;
	float	*	h_b_dataTimes;
	float	*	h_c_dataTimes;
	float	*	h_symmetry_dataTimes;

	float	*	d_data;
	int		*	d_dataSizes;
	float	*	d_a_dataTimes;
	float	*	d_b_dataTimes;
	float	*	d_c_dataTimes;
	float	*	d_symmetry_dataTimes;

	size_t amountOfIteration = (size_t)std::ceilf((float)in_nPts / (float)nPtsLimiter);;

	std::ofstream outFileStream;
	outFileStream.open(in_outPath);

	for (size_t i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
		{
			h_dataTimes = (float*)malloc((in_nPts - nPtsLimiter * i) * sizeof(float));

			h_a_dataTimes			= (float*)malloc((in_nPts - nPtsLimiter * i) * sizeof(float));
			h_b_dataTimes			= (float*)malloc((in_nPts - nPtsLimiter * i) * sizeof(float));
			h_c_dataTimes			= (float*)malloc((in_nPts - nPtsLimiter * i) * sizeof(float));
			h_symmetry_dataTimes	= (float*)malloc((in_nPts - nPtsLimiter * i) * sizeof(float));

			slice(globalParamValues, nPtsLimiter * i, in_nPts, h_dataTimes);
			slice(aParamValues, nPtsLimiter * i, in_nPts, h_a_dataTimes);
			slice(bParamValues, nPtsLimiter * i, in_nPts, h_b_dataTimes);
			slice(cParamValues, nPtsLimiter * i, in_nPts, h_c_dataTimes);
			slice(symmetryParamValues, nPtsLimiter * i, in_nPts, h_symmetry_dataTimes);
			nPtsLimiter = in_nPts - (nPtsLimiter * i);
		}
		else
		{
			h_dataTimes = (float*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(float));

			h_a_dataTimes			= (float*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(float));
			h_b_dataTimes			= (float*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(float));
			h_c_dataTimes			= (float*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(float));
			h_symmetry_dataTimes	= (float*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(float));

			slice(globalParamValues, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_dataTimes);
			slice(aParamValues, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_a_dataTimes);
			slice(bParamValues, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_b_dataTimes);
			slice(cParamValues, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_c_dataTimes);
			slice(symmetryParamValues, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_symmetry_dataTimes);
		}


		h_data = (float*)malloc(nPtsLimiter * amountOfTPoints * sizeof(float));
		h_dataSizes = (int*)malloc(nPtsLimiter * sizeof(int));

		cudaMalloc((void**)&d_data, nPtsLimiter * amountOfTPoints * sizeof(float));
		cudaMalloc((void**)&d_dataSizes, nPtsLimiter * sizeof(int));
		cudaMalloc((void**)&d_a_dataTimes, nPtsLimiter * sizeof(float));
		cudaMalloc((void**)&d_b_dataTimes, nPtsLimiter * sizeof(float));
		cudaMalloc((void**)&d_c_dataTimes, nPtsLimiter * sizeof(float));
		cudaMalloc((void**)&d_symmetry_dataTimes, nPtsLimiter * sizeof(float));

		cudaMemcpy(d_a_dataTimes, h_a_dataTimes, nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaMemcpy(d_b_dataTimes, h_b_dataTimes, nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaMemcpy(d_c_dataTimes, h_c_dataTimes, nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaMemcpy(d_symmetry_dataTimes, h_symmetry_dataTimes, nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

		int blockSize;
		int minGridSize;
		int gridSize;

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bifuractionKernel, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;



		//Call CUDA func
		bifuractionKernel<<<gridSize, blockSize >>>(		d_a_dataTimes,
															d_b_dataTimes,
															d_c_dataTimes,
															d_symmetry_dataTimes,
															nPtsLimiter,
															in_tMax,
															in_h,
															in_initialCondition1,
															in_initialCondition2,
															in_initialCondition3,
															in_nValue,
															in_prePeakFinderSliceK,
															d_data,
															d_dataSizes,
															PEAKFINDER_MODE);


		cudaMemcpy(h_data, d_data, amountOfTPoints * nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaMemcpy(h_dataSizes, d_dataSizes, nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		cudaFree(d_data);
		cudaFree(d_dataSizes);
		//cudaFree(d_dataTimes);
		cudaFree(d_a_dataTimes);
		cudaFree(d_b_dataTimes);
		cudaFree(d_c_dataTimes);
		cudaFree(d_symmetry_dataTimes);

		for (size_t i = 0; i < nPtsLimiter; ++i)
			for (size_t j = 0; j < h_dataSizes[i]; ++j)
				if (outFileStream.is_open())
				{
					outFileStream << h_dataTimes[i] << ", " << h_data[i * amountOfTPoints + j] << '\n';
				}
				else
				{
					std::cout << "\nOutput file open error" << std::endl;
					exit(1);
				}

		std::free(h_data);
		std::free(h_dataSizes);
		std::free(h_dataTimes);
		std::free(h_a_dataTimes);
		std::free(h_b_dataTimes);
		std::free(h_c_dataTimes);
		std::free(h_symmetry_dataTimes);

		if (in_debug)
			std::cout << "       " << std::setprecision(3) << (100.0f / (float)amountOfIteration) * (i + 1) << "%\n";

		progress.store((100.0f / (float)amountOfIteration) * (i + 1), std::memory_order_seq_cst);
	}

	if (in_debug)
	{
		if (amountOfIteration != 1)
			std::cout << "       " << "100%\n";
		std::cout << '\n';
	}

	std::free(globalParamValues);

	progress.store(100, std::memory_order_seq_cst);

	outFileStream.close();

	return;
}



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
	std::atomic<int>& progress)
{
	std::ofstream outFileStream;
	outFileStream.open(in_outPath);
	outFileStream << in_paramValues1 << ", " << in_paramValues2 << "\n" << in_paramValues3 << ", " << in_paramValues4 << "\n";

	size_t amountOfTPoints = in_tMax / in_h;

	float* aParamValues = nullptr;
	float* bParamValues = nullptr;
	float* cParamValues = nullptr;
	float* symmetryParamValues = nullptr;

	aParamValues = (float*)malloc(sizeof(float) * in_nPts * in_nPts);
	bParamValues = (float*)malloc(sizeof(float) * in_nPts * in_nPts);
	cParamValues = (float*)malloc(sizeof(float) * in_nPts * in_nPts);
	symmetryParamValues = (float*)malloc(sizeof(float) * in_nPts * in_nPts);

	getParamsAndSymmetry(aParamValues, bParamValues, cParamValues, symmetryParamValues,
		in_paramA, in_paramB, in_paramC, 0.5f,
		in_paramValues1, in_paramValues2, in_mode1, 
		in_paramValues3, in_paramValues4, in_mode2, 
		in_nPts);

	size_t freeMemory;
	size_t totalMemory;

	cudaMemGetInfo(&freeMemory, &totalMemory);

	freeMemory *= in_memoryLimit;

	float maxMemoryLimit = sizeof(float) * ((in_tMax / in_h) + 4) + sizeof(int);

	size_t nPtsLimiter = freeMemory / maxMemoryLimit;

	if (nPtsLimiter <= 0)
	{
		if (in_debug)
			std::cout << "\nVery low memory size. Increase the MEMORY_LIMIT!" << "\n";
		exit(1);
	}

	int*   h_kdeResult;
	float* h_data;
	float* h_a_dataTimes;
	float* h_b_dataTimes;
	float* h_c_dataTimes;
	float* h_symmetry_dataTimes;

	int*   d_kdeResult;
	float* d_data;
	float* d_a_dataTimes;
	float* d_b_dataTimes;
	float* d_c_dataTimes;
	float* d_symmetry_dataTimes;

	size_t amountOfIteration = (size_t)std::ceilf((float)(in_nPts * in_nPts) / (float)nPtsLimiter);;

	int stringCounter = 0;

	for (size_t i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
		{
			h_kdeResult   = (int*)malloc(((in_nPts * in_nPts) - nPtsLimiter * i) * sizeof(int));
			h_a_dataTimes = (float*)malloc(((in_nPts * in_nPts) - nPtsLimiter * i) * sizeof(float));
			h_b_dataTimes = (float*)malloc(((in_nPts * in_nPts) - nPtsLimiter * i) * sizeof(float));
			h_c_dataTimes = (float*)malloc(((in_nPts * in_nPts) - nPtsLimiter * i) * sizeof(float));
			h_symmetry_dataTimes = (float*)malloc(((in_nPts * in_nPts) - nPtsLimiter * i) * sizeof(float));

			slice(aParamValues, nPtsLimiter * i, (in_nPts * in_nPts), h_a_dataTimes);
			slice(bParamValues, nPtsLimiter * i, (in_nPts * in_nPts), h_b_dataTimes);
			slice(cParamValues, nPtsLimiter * i, (in_nPts * in_nPts), h_c_dataTimes);
			slice(symmetryParamValues, nPtsLimiter * i, (in_nPts * in_nPts), h_symmetry_dataTimes);
			nPtsLimiter = (in_nPts * in_nPts) - (nPtsLimiter * i);
		}
		else
		{
			h_kdeResult = (int*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(int));
			h_a_dataTimes = (float*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(float));
			h_b_dataTimes = (float*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(float));
			h_c_dataTimes = (float*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(float));
			h_symmetry_dataTimes = (float*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(float));

			slice(aParamValues, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_a_dataTimes);
			slice(bParamValues, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_b_dataTimes);
			slice(cParamValues, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_c_dataTimes);
			slice(symmetryParamValues, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_symmetry_dataTimes);
		}


		h_data = (float*)malloc(nPtsLimiter * amountOfTPoints * sizeof(float));

		cudaMalloc((void**)&d_kdeResult, nPtsLimiter * sizeof(int));
		cudaMalloc((void**)&d_data, nPtsLimiter * amountOfTPoints * sizeof(float));
		cudaMalloc((void**)&d_a_dataTimes, nPtsLimiter * sizeof(float));
		cudaMalloc((void**)&d_b_dataTimes, nPtsLimiter * sizeof(float));
		cudaMalloc((void**)&d_c_dataTimes, nPtsLimiter * sizeof(float));
		cudaMalloc((void**)&d_symmetry_dataTimes, nPtsLimiter * sizeof(float));

		cudaMemcpy(d_a_dataTimes, h_a_dataTimes, nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaMemcpy(d_b_dataTimes, h_b_dataTimes, nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaMemcpy(d_c_dataTimes, h_c_dataTimes, nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaMemcpy(d_symmetry_dataTimes, h_symmetry_dataTimes, nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

		int blockSize;
		int minGridSize;
		int gridSize;

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bifuractionKernel, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;



		//Call CUDA func
		bifuractionKernel << <gridSize, blockSize >> > (d_a_dataTimes,
			d_b_dataTimes,
			d_c_dataTimes,
			d_symmetry_dataTimes,
			nPtsLimiter,
			in_tMax,
			in_h,
			in_initialCondition1,
			in_initialCondition2,
			in_initialCondition3,
			in_nValue,
			in_prePeakFinderSliceK,
			d_data,
			d_kdeResult,
			KDE_MODE,
			in_kdeSampling,
			in_kdeSamplesInterval1,
			in_kdeSamplesInterval2,
			in_kdeSamplesSmooth);


		//cudaMemcpy(h_data, d_data, amountOfTPoints * nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaMemcpy(h_kdeResult, d_kdeResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		cudaFree(d_data);
		cudaFree(d_kdeResult);
		cudaFree(d_a_dataTimes);
		cudaFree(d_b_dataTimes);
		cudaFree(d_c_dataTimes);
		cudaFree(d_symmetry_dataTimes);

		for (size_t i = 0; i < nPtsLimiter; ++i)
			if (outFileStream.is_open())
			{
				if (stringCounter != 0)
					outFileStream << ", ";
				if (stringCounter == in_nPts)
				{
					outFileStream << "\n";
					stringCounter = 0;
				}
				outFileStream << h_kdeResult[i];
				++stringCounter;
			}
			else
			{
				std::cout << "\nOutput file open error" << std::endl;
				exit(1);
			}

		std::free(h_kdeResult);
		std::free(h_data);
		std::free(h_a_dataTimes);
		std::free(h_b_dataTimes);
		std::free(h_c_dataTimes);
		std::free(h_symmetry_dataTimes);

		if (in_debug)
			std::cout << "       " << std::setprecision(3) << (100.0f / (float)amountOfIteration) * (i + 1) << "%\n";

		progress.store((100.0f / (float)amountOfIteration) * (i + 1), std::memory_order_seq_cst);
	}

	if (in_debug)
	{
		if (amountOfIteration != 1)
			std::cout << "       " << "100%\n";
		std::cout << '\n';
	}

	progress.store(100, std::memory_order_seq_cst);

	outFileStream.close();

	return;
}



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
									int						in_kdeSampling,
									float					in_kdeSamplesInterval1,
									float					in_kdeSamplesInterval2,
									float					in_kdeSmoothH)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= in_nPts)
		return;

	size_t amountOfTPoints = in_TMax / in_h;

	float localH1 = in_h * in_symmetry[idx];
	float localH2 = in_h * (1 - in_symmetry[idx]);

	float x[3]{ in_initialCondition1, in_initialCondition2, in_initialCondition3 };

	for (size_t i = 0; i < amountOfTPoints; ++i)
	{
		in_data[idx * amountOfTPoints + i] = x[in_nValue];

		x[0] = x[0] + localH1 * (-x[1] - x[2]);
		x[1] = (x[1] + localH1 * (x[0])) / (1 - in_paramA[idx] * localH1);
		x[2] = (x[2] + localH1 * in_paramB[idx]) / (1 - localH1 * (x[0] - in_paramC[idx]));

		x[2] = x[2] + localH2 * (in_paramB[idx] + x[2] * (x[0] - in_paramC[idx]));
		x[1] = x[1] + localH2 * (x[0] + in_paramA[idx] * x[1]);
		x[0] = x[0] + localH2 * (-x[1] - x[2]);
		if (resultMode == KDE_MODE && abs(x[in_nValue]) > 10000.0f)
		{
			in_dataSizes[idx] = 0;
			return;
		}
	}

	// Here is the choice of method: KDE or peakFinder
	// TODO add switch on method of result
	// WARNING!!! THIS METHOD MUST TO FILL dataSizes[] (IF NEEDED)!!!

	int outSize = 0;

	switch (resultMode)
	{
	case PEAKFINDER_MODE:
		peakFinder(idx, in_prePeakFinderSliceK, amountOfTPoints, in_data, in_dataSizes, in_data);
		break;
	case KDE_MODE:
		outSize = peakFinder(idx, in_prePeakFinderSliceK, amountOfTPoints, in_data, in_dataSizes, in_data);
		kdeMethod(idx, in_data, in_dataSizes, in_kdeSampling, outSize, in_kdeSamplesInterval1, in_kdeSamplesInterval2, amountOfTPoints, in_kdeSmoothH);
		break;
	}
}



__device__ int peakFinder(int idx, float prePeakFinder, size_t amountOfTPoints, float* in_data, int* out_dataSizes, float* out_data)
{
	int _outSize = 0;
	for (size_t i = 1 + prePeakFinder * amountOfTPoints; i < amountOfTPoints - 1; ++i)
	{
		if (in_data[idx * amountOfTPoints + i] > in_data[idx * amountOfTPoints + i - 1] && in_data[idx * amountOfTPoints + i] > in_data[idx * amountOfTPoints + i + 1])
		{
			out_data[idx * amountOfTPoints + _outSize] = in_data[idx * amountOfTPoints + i];
			++_outSize;
		}
		else if (in_data[idx * amountOfTPoints + i] > in_data[idx * amountOfTPoints + i - 1] && in_data[idx * amountOfTPoints + i] == in_data[idx * amountOfTPoints + i + 1])
		{
			for (size_t k = i; k < amountOfTPoints - 1; ++k)
			{
				if (in_data[idx * amountOfTPoints + k] < in_data[idx * amountOfTPoints + k + 1])
				{
					break;
					i = k;
				}
				if (in_data[idx * amountOfTPoints + k] == in_data[idx * amountOfTPoints + k + 1])
					continue;
				if (in_data[idx * amountOfTPoints + k] > in_data[idx * amountOfTPoints + k + 1])
				{
					out_data[idx * amountOfTPoints + _outSize] = in_data[idx * amountOfTPoints + k];
					++_outSize;
					i = k + 1;
					break;
				}
			}
		}
	}

	out_dataSizes[idx] = _outSize;
	return _outSize;
}



__device__ void kdeMethod(int idx, 
	float* data,
	int* kdeResult,
	int kdeSampling, 
	int _outSize, 
	float kdeSamplesInterval1, 
	float kdeSamplesInterval2,
	size_t amountOfTPoints,
	float kdeSmoothH)
{
	float k1 = kdeSampling * _outSize;
	float k2 = (kdeSamplesInterval2 - kdeSamplesInterval1) / (k1 - 1);
	float delt = 0;
	float prevPrevData2 = 0;
	float prevData2 = 0;
	float data2 = 0;
	float memoryData2 = 0;
	bool strangePeak = false;
	int resultKde = 0;

	if (_outSize == 0)
	{
		kdeResult[idx] = 0;
		return;
	}

	if (_outSize == 1)
	{
		kdeResult[idx] = 1;
		return;
	}

	if (_outSize == 2)
	{
		kdeResult[idx] = 1;
		return;
	}

	for (int w = 0; w < k1 - 1; ++w)
	{
		delt = w * k2 + kdeSamplesInterval1;
		prevPrevData2 = prevData2;
		prevData2 = data2;
		data2 = 0;
		for (int m = 0; m < _outSize; ++m)
		{
			float tempData = (data[idx * amountOfTPoints + m] - delt) / kdeSmoothH;
			data2 += expf(-((tempData * tempData) / 2));
		}
		// Íàéòè çäåñü - ÿâëÿåòñÿ ëè çäåñü data2 ïèêîì èëè íåò. Åñëè äà - èíêðåìèðóåì resultKde
		if (w < 2)
			continue;

		if (strangePeak)
		{
			if (prevData2 == data2)
				continue;
			else if (prevData2 < data2)
			{
				strangePeak = false;
				continue;
			}
			else if (prevData2 > data2)
			{
				strangePeak = false;
				++resultKde;
				continue;
			}
		}
		else if (prevData2 > prevPrevData2 && prevData2 > data2)
		{
			++resultKde;
			continue;
		}
		else if (prevData2 > prevPrevData2 && prevData2 == data2)
		{
			strangePeak = true;
			memoryData2 = prevData2;
			continue;
		}
	}
	if (prevData2 < data2)
	{
		++resultKde;
	}
	kdeResult[idx] = resultKde;
	return;
}



template <class T1, class T2>
__host__ void linspace(T1 a, T1 b, int amount, T2* out, int startIndex)
{
	if (amount <= 0)
		throw std::invalid_argument("linspace error. amount <= 0");
	if (amount == 1)
	{
		out[0] = a;
		return;
	}

	float step = (b - a) / (amount - 1);
	for (size_t i = 0; i < amount; ++i)
		out[startIndex + i] = a + i * step;

	return;
}



__host__ void getParamsAndSymmetry(float* a, float* b, float* c, float* symmetry,
	float in_a, float in_b, float in_c, float in_symmetry,
	float startInterval1, float finishInteraval1, int mode1,
	int nPts)
{
	if (mode1 == PARAM_A)
		linspace(startInterval1, finishInteraval1, nPts, a);
	else
		for (int i = 0; i < nPts; ++i)
			a[i] = in_a;

	if (mode1 == PARAM_B)
		linspace(startInterval1, finishInteraval1, nPts, b);
	else
		for (int i = 0; i < nPts; ++i)
			b[i] = in_b;

	if (mode1 == PARAM_C)
		linspace(startInterval1, finishInteraval1, nPts, c);
	else
		for (int i = 0; i < nPts; ++i)
			c[i] = in_c;

	if (mode1 == SYMMETRY)
		linspace(startInterval1, finishInteraval1, nPts, symmetry);
	else
		for (int i = 0; i < nPts; ++i)
			symmetry[i] = in_symmetry;
}



__host__ void getParamsAndSymmetry(float* a, float* b, float* c, float* symmetry,
	float in_a, float in_b, float in_c, float in_symmetry,
	float startInterval1, float finishInteraval1, int mode1,
	float startInterval2, float finishInteraval2, int mode2,
	int nPts)
{
	float* tempParams = new float[nPts];
	linspace(startInterval2, finishInteraval2, nPts, tempParams);

	for (int i = 0; i < nPts * nPts; ++i)
	{
		a[i] = in_a;
		b[i] = in_b;
		c[i] = in_c;
		symmetry[i] = in_symmetry;
	}

	if (mode1 == PARAM_A)
	{
		for (int i = 0; i < nPts; ++i)
		{
			linspace(startInterval1, finishInteraval1, nPts, a, i * nPts);
			switch (mode2)
			{
			case PARAM_A:
				break;
			case PARAM_B:
				for (int j = 0; j < nPts; ++j)
					b[nPts * i + j] = tempParams[i];
				break;
			case PARAM_C:
				for (int j = 0; j < nPts; ++j)
					c[nPts * i + j] = tempParams[i];
				break;
			case SYMMETRY:
				for (int j = 0; j < nPts; ++j)
					symmetry[nPts * i + j] = tempParams[i];
				break;
			}
		}
	}

	if (mode1 == PARAM_B)
	{
		for (int i = 0; i < nPts; ++i)
		{
			linspace(startInterval1, finishInteraval1, nPts, b, i * nPts);
			switch (mode2)
			{
			case PARAM_A:
				for (int j = 0; j < nPts; ++j)
					a[nPts * i + j] = tempParams[i];
				break;
			case PARAM_B:
				break;
			case PARAM_C:
				for (int j = 0; j < nPts; ++j)
					c[nPts * i + j] = tempParams[i];
				break;
			case SYMMETRY:
				for (int j = 0; j < nPts; ++j)
					symmetry[nPts * i + j] = tempParams[i];
				break;
			}
		}
	}

	if (mode1 == PARAM_C)
	{
		for (int i = 0; i < nPts; ++i)
		{
			linspace(startInterval1, finishInteraval1, nPts, c, i * nPts);
			switch (mode2)
			{
			case PARAM_A:
				for (int j = 0; j < nPts; ++j)
					a[nPts * i + j] = tempParams[i];
				break;
			case PARAM_B:
				for (int j = 0; j < nPts; ++j)
					b[nPts * i + j] = tempParams[i];
				break;
			case PARAM_C:
				break;
			case SYMMETRY:
				for (int j = 0; j < nPts; ++j)
					symmetry[nPts * i + j] = tempParams[i];
				break;
			}
		}
	}

	if (mode1 == SYMMETRY)
	{
		for (int i = 0; i < nPts; ++i)
		{
			linspace(startInterval1, finishInteraval1, nPts, symmetry, i * nPts);
			switch (mode2)
			{
			case PARAM_A:
				for (int j = 0; j < nPts; ++j)
					a[nPts * i + j] = tempParams[i];
				break;
			case PARAM_B:
				for (int j = 0; j < nPts; ++j)
					b[nPts * i + j] = tempParams[i];
				break;
			case PARAM_C:
				for (int j = 0; j < nPts; ++j)
					c[nPts * i + j] = tempParams[i];
				break;
			case SYMMETRY:
				break;
			}
		}
	}
	return;
}



template <class T>
__host__ void slice(T* in, int a, int b, T* out)
{
	if (b - a < 0)
		throw std::invalid_argument("slice error. b < a");
	for (size_t i = 0; i < b - a; ++i)
		out[i] = in[a + i];
}


