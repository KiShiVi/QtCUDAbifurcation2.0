#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "bifurcationKernel.cuh"

__host__ void bifurcation1D(
	double					in_tMax,
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
	std::atomic<int>& progress)
{
	size_t amountOfTPoints = in_tMax / in_h / in_prescaller;

	double* globalParamValues = nullptr;
	globalParamValues = (double*)malloc(sizeof(double) * in_nPts);
	linspace(in_paramValues1, in_paramValues2, in_nPts, globalParamValues);

	size_t freeMemory;
	size_t totalMemory;

	cudaMemGetInfo(&freeMemory, &totalMemory);

	freeMemory *= in_memoryLimit * 0.95;

	double maxMemoryLimit = sizeof(double) * (amountOfTPoints + 2 + in_amountOfParams) + sizeof(int);

	size_t nPtsLimiter = freeMemory / maxMemoryLimit;
	//ne takaya yzh huita ebanaya
	if (nPtsLimiter <= 0)
	{
		if (in_debug)
			std::cout << "\nVery low memory size. Increase the MEMORY_LIMIT!" << "\n";
		exit(1);
	}

	float* h_data;
	int* h_dataSizes;
	double* h_dataTimes;

	float* d_data;
	int* d_dataSizes;
	double* d_dataTimes;
	double* d_params;
	double* d_initialConditions;

	cudaMalloc((void**)& d_params, in_amountOfParams * sizeof(double));
	cudaMalloc((void**)& d_initialConditions, in_amountOfParams * sizeof(double));
	cudaMemcpy(d_params, in_params, in_amountOfParams * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(d_initialConditions, in_initialConditions, in_amountOfParams * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	size_t amountOfIteration = (size_t)std::ceilf((double)in_nPts / (double)nPtsLimiter);;

	std::ofstream outFileStream;
	outFileStream.open(in_outPath);

	for (size_t i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
		{
			h_dataTimes = (double*)malloc((in_nPts - nPtsLimiter * i) * sizeof(double));

			slice(globalParamValues, nPtsLimiter * i, in_nPts, h_dataTimes);
			nPtsLimiter = in_nPts - (nPtsLimiter * i);
		}
		else
		{
			h_dataTimes = (double*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(double));
			slice(globalParamValues, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_dataTimes);
		}


		h_data = (float*)malloc(nPtsLimiter * amountOfTPoints * sizeof(float));
		h_dataSizes = (int*)malloc(nPtsLimiter * sizeof(int));

		cudaMalloc((void**)& d_data, nPtsLimiter * amountOfTPoints * sizeof(float));
		cudaMalloc((void**)& d_dataSizes, nPtsLimiter * sizeof(int));
		cudaMalloc((void**)& d_dataTimes, nPtsLimiter * sizeof(double));

		cudaMemcpy(d_dataTimes, h_dataTimes, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

		int blockSize;
		int minGridSize;
		int gridSize;

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bifuractionKernel, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;



		//Call CUDA func
		bifuractionKernel << <gridSize, blockSize >> > (nPtsLimiter,
			in_tMax,
			in_h,
			d_initialConditions,
			in_nValue,
			in_prePeakFinderSliceK,
			d_data,
			d_dataSizes,
			PEAKFINDER_MODE,
			in_thresholdValueOfMaxSignalValue,
			in_amountOfParams,
			in_discreteModelMode,
			in_prescaller,
			d_params,
			d_dataTimes,
			in_mode);


		cudaMemcpy(h_data, d_data, amountOfTPoints * nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaMemcpy(h_dataSizes, d_dataSizes, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		cudaFree(d_data);
		cudaFree(d_dataSizes);
		cudaFree(d_dataTimes);

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

		if (in_debug)
			std::cout << "       " << std::setprecision(3) << (100.0f / (double)amountOfIteration) * (i + 1) << "%\n";

		progress.store((100.0f / (double)amountOfIteration) * (i + 1), std::memory_order_seq_cst);
	}

	if (in_debug)
	{
		if (amountOfIteration != 1)
			std::cout << "       " << "100%\n";
		std::cout << '\n';
	}

	cudaFree(d_params);
	cudaFree(d_initialConditions);
	std::free(globalParamValues);

	progress.store(100, std::memory_order_seq_cst);

	outFileStream.close();

	return;
}



__host__ void bifurcation2D(
	double					in_tMax,
	int					in_nPts,
	double				in_h,
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
	std::atomic<int> & progress)
{
	std::ofstream outFileStream;
	outFileStream.open(in_outPath);
	outFileStream << in_paramValues1 << ", " << in_paramValues2 << "\n" << in_paramValues3 << ", " << in_paramValues4 << "\n";

	size_t amountOfTPoints = in_tMax / in_h / in_prescaller;

	double* paramValues1 = nullptr;
	double* paramValues2 = nullptr;

	paramValues1 = (double*)malloc(sizeof(double) * in_nPts * in_nPts);
	paramValues2 = (double*)malloc(sizeof(double) * in_nPts * in_nPts);

	getParamsAndSymmetry2D(paramValues1, paramValues2,
		in_paramValues1, in_paramValues2,
		in_paramValues3, in_paramValues4,
		in_nPts);

	size_t freeMemory;
	size_t totalMemory;

	cudaMemGetInfo(&freeMemory, &totalMemory);
	//freeMemory = 7472152576;
	freeMemory *= in_memoryLimit * 0.95;

	double maxMemoryLimit = sizeof(double) * (amountOfTPoints + 2 + in_amountOfParams) + sizeof(int);

	size_t nPtsLimiter = freeMemory / maxMemoryLimit;

	if (nPtsLimiter <= 0)
	{
		if (in_debug)
			std::cout << "\nVery low memory size. Increase the MEMORY_LIMIT!" << "\n";
		exit(1);
	}

	int* h_kdeResult;
	float* h_data;
	double* h_paramValues1;
	double* h_paramValues2;

	double* d_params;
	int* d_kdeResult;
	float* d_data;
	double* d_paramValues1;
	double* d_paramValues2;
	double* d_initialConditions;


	cudaMalloc((void**)& d_params, in_amountOfParams * sizeof(double));
	cudaMalloc((void**)& d_initialConditions, in_amountOfParams * sizeof(double));

	cudaMemcpy(d_params, in_params, in_amountOfParams * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(d_initialConditions, in_initialConditions, in_amountOfParams * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	size_t amountOfIteration = (size_t)std::ceilf((double)(in_nPts * in_nPts) / (double)nPtsLimiter);

	int stringCounter = 0;

	for (size_t i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
		{
			h_kdeResult = (int*)malloc(((in_nPts * in_nPts) - nPtsLimiter * i) * sizeof(int));
			h_paramValues1 = (double*)malloc(((in_nPts * in_nPts) - nPtsLimiter * i) * sizeof(double));
			h_paramValues2 = (double*)malloc(((in_nPts * in_nPts) - nPtsLimiter * i) * sizeof(double));

			slice(paramValues1, nPtsLimiter * i, (in_nPts * in_nPts), h_paramValues1);
			slice(paramValues2, nPtsLimiter * i, (in_nPts * in_nPts), h_paramValues2);
			nPtsLimiter = (in_nPts * in_nPts) - (nPtsLimiter * i);
		}
		else
		{
			h_kdeResult = (int*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(int));
			h_paramValues1 = (double*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(double));
			h_paramValues2 = (double*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(double));

			slice(paramValues1, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_paramValues1);
			slice(paramValues2, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_paramValues2);
		}


		h_data = (float*)malloc(nPtsLimiter * amountOfTPoints * sizeof(float));

		cudaMalloc((void**)& d_kdeResult, nPtsLimiter * sizeof(int));
		cudaMalloc((void**)& d_data, nPtsLimiter * amountOfTPoints * sizeof(double));
		cudaMalloc((void**)& d_paramValues1, nPtsLimiter * sizeof(double));
		cudaMalloc((void**)& d_paramValues2, nPtsLimiter * sizeof(double));

		cudaMemcpy(d_paramValues1, h_paramValues1, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaMemcpy(d_paramValues2, h_paramValues2, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

		int blockSize;
		int minGridSize;
		int gridSize;

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bifuractionKernel, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		//Call CUDA func
		bifuractionKernel << <gridSize, blockSize >> > (
			nPtsLimiter,
			in_tMax,
			in_h,
			d_initialConditions,
			in_nValue,
			in_prePeakFinderSliceK,
			d_data,
			d_kdeResult,
			KDE_MODE,
			in_thresholdValueOfMaxSignalValue,
			in_amountOfParams,
			in_discreteModelMode,
			in_prescaller,
			d_params,
			d_paramValues1,
			in_mode1,
			d_paramValues2,
			in_mode2,
			nullptr,
			0,
			in_kdeSampling,
			in_kdeSamplesInterval1,
			in_kdeSamplesInterval2,
			in_kdeSamplesSmooth);


		//cudaMemcpy(h_data, d_data, amountOfTPoints * nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaMemcpy(h_kdeResult, d_kdeResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		cudaFree(d_data);
		cudaFree(d_kdeResult);
		cudaFree(d_paramValues1);
		cudaFree(d_paramValues2);

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
		std::free(h_paramValues1);
		std::free(h_paramValues2);

		if (in_debug)
			std::cout << "       " << std::setprecision(3) << (100.0f / (double)amountOfIteration) * (i + 1) << "%\n";

		progress.store((100.0f / (double)amountOfIteration) * (i + 1), std::memory_order_seq_cst);
	}

	cudaFree(d_params);
	cudaFree(d_initialConditions);

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
	std::atomic<int> & progress)
{
	std::ofstream outFileStream;
	outFileStream.open(in_outPath);
	outFileStream << in_paramValues1 << ", " << in_paramValues2 << "\n" << in_paramValues3 << ", " << in_paramValues4 << "\n" << in_paramValues5 << ", " << in_paramValues6 << "\n";

	size_t amountOfTPoints = in_tMax / in_h / in_prescaller;

	double* paramValues1 = nullptr;
	double* paramValues2 = nullptr;
	double* paramValues3 = nullptr;

	paramValues1 = (double*)malloc(sizeof(double) * in_nPts * in_nPts * in_nPts);
	paramValues2 = (double*)malloc(sizeof(double) * in_nPts * in_nPts * in_nPts);
	paramValues3 = (double*)malloc(sizeof(double) * in_nPts * in_nPts * in_nPts);

	getParamsAndSymmetry3D(paramValues1, paramValues2, paramValues3,
		in_paramValues1, in_paramValues2,
		in_paramValues3, in_paramValues4,
		in_paramValues5, in_paramValues6,
		in_nPts);

	size_t freeMemory;
	size_t totalMemory;

	cudaMemGetInfo(&freeMemory, &totalMemory);
	//freeMemory = 7472152576;
	freeMemory *= in_memoryLimit * 0.95;

	double maxMemoryLimit = sizeof(double) * ((amountOfTPoints)+3 + in_amountOfParams) + sizeof(int);

	size_t nPtsLimiter = freeMemory / maxMemoryLimit;

	if (nPtsLimiter <= 0)
	{
		if (in_debug)
			std::cout << "\nVery low memory size. Increase the MEMORY_LIMIT!" << "\n";
		exit(1);
	}

	int* h_kdeResult;
	float* h_data;
	double* h_paramValues1;
	double* h_paramValues2;
	double* h_paramValues3;

	double* d_params;
	int* d_kdeResult;
	float* d_data;
	double* d_paramValues1;
	double* d_paramValues2;
	double* d_paramValues3;
	double* d_initialConditions;


	cudaMalloc((void**)& d_params, in_amountOfParams * sizeof(double));
	cudaMalloc((void**)& d_initialConditions, in_amountOfParams * sizeof(double));

	cudaMemcpy(d_params, in_params, in_amountOfParams * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(d_initialConditions, in_initialConditions, in_amountOfParams * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	size_t amountOfIteration = (size_t)std::ceilf((double)(in_nPts * in_nPts * in_nPts) / (double)nPtsLimiter);

	int stringCounter = 0;

	for (size_t i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
		{
			h_kdeResult = (int*)malloc(((in_nPts * in_nPts * in_nPts) - nPtsLimiter * i) * sizeof(int));
			h_paramValues1 = (double*)malloc(((in_nPts * in_nPts * in_nPts) - nPtsLimiter * i) * sizeof(double));
			h_paramValues2 = (double*)malloc(((in_nPts * in_nPts * in_nPts) - nPtsLimiter * i) * sizeof(double));
			h_paramValues3 = (double*)malloc(((in_nPts * in_nPts * in_nPts) - nPtsLimiter * i) * sizeof(double));

			slice(paramValues1, nPtsLimiter * i, (in_nPts * in_nPts * in_nPts), h_paramValues1);
			slice(paramValues2, nPtsLimiter * i, (in_nPts * in_nPts * in_nPts), h_paramValues2);
			slice(paramValues3, nPtsLimiter * i, (in_nPts * in_nPts * in_nPts), h_paramValues3);
			nPtsLimiter = (in_nPts * in_nPts * in_nPts) - (nPtsLimiter * i);
		}
		else
		{
			h_kdeResult = (int*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(int));
			h_paramValues1 = (double*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(double));
			h_paramValues2 = (double*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(double));
			h_paramValues3 = (double*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(double));

			slice(paramValues1, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_paramValues1);
			slice(paramValues2, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_paramValues2);
			slice(paramValues3, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_paramValues3);
		}


		h_data = (float*)malloc(nPtsLimiter * amountOfTPoints * sizeof(float));

		cudaMalloc((void**)& d_kdeResult, nPtsLimiter * sizeof(int));
		cudaMalloc((void**)& d_data, nPtsLimiter * amountOfTPoints * sizeof(double));
		cudaMalloc((void**)& d_paramValues1, nPtsLimiter * sizeof(double));
		cudaMalloc((void**)& d_paramValues2, nPtsLimiter * sizeof(double));
		cudaMalloc((void**)& d_paramValues3, nPtsLimiter * sizeof(double));

		cudaMemcpy(d_paramValues1, h_paramValues1, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaMemcpy(d_paramValues2, h_paramValues2, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaMemcpy(d_paramValues3, h_paramValues3, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

		int blockSize;
		int minGridSize;
		int gridSize;

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bifuractionKernel, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		//Call CUDA func
		bifuractionKernel << <gridSize, blockSize >> > (
			nPtsLimiter,
			in_tMax,
			in_h,
			d_initialConditions,
			in_nValue,
			in_prePeakFinderSliceK,
			d_data,
			d_kdeResult,
			KDE_MODE,
			in_thresholdValueOfMaxSignalValue,
			in_amountOfParams,
			in_discreteModelMode,
			in_prescaller,
			d_params,
			d_paramValues1,
			in_mode1,
			d_paramValues2,
			in_mode2,
			d_paramValues3,
			in_mode3,
			in_kdeSampling,
			in_kdeSamplesInterval1,
			in_kdeSamplesInterval2,
			in_kdeSamplesSmooth);


		//cudaMemcpy(h_data, d_data, amountOfTPoints * nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaMemcpy(h_kdeResult, d_kdeResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		cudaFree(d_data);
		cudaFree(d_kdeResult);
		cudaFree(d_paramValues1);
		cudaFree(d_paramValues2);
		cudaFree(d_paramValues3);

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
		std::free(h_paramValues1);
		std::free(h_paramValues2);
		std::free(h_paramValues3);

		if (in_debug)
			std::cout << "       " << std::setprecision(3) << (100.0f / (double)amountOfIteration) * (i + 1) << "%\n";

		progress.store((100.0f / (double)amountOfIteration) * (i + 1), std::memory_order_seq_cst);
	}

	cudaFree(d_params);
	cudaFree(d_initialConditions);

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



__global__ void bifuractionKernel(
	int in_nPts,
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
	double* in_paramValues2,
	int in_mode2,
	double* in_paramValues3,
	int in_mode3,
	int in_kdeSampling,
	float in_kdeSamplesInterval1,
	float in_kdeSamplesInterval2,
	float in_kdeSmoothH
)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= in_nPts)
		return;

	size_t amountOfTPoints = in_TMax / in_h / in_prescaller;
	size_t amountOfSkipPoints = in_prePeakFinderSliceK / in_h;
	size_t index = amountOfTPoints * idx;
	// Change to dynamic / KISH
	//double x[3]{ in_initialConditions[0], in_initialConditions[1], in_initialConditions[2] };
	double x[4]{ in_initialConditions[0], in_initialConditions[1], in_initialConditions[2], in_initialConditions[3] };


	double* localParam = new double[in_amountOfParams];
	for (int i = 0; i < in_amountOfParams; ++i)
		localParam[i] = in_params[i];

	localParam[in_mode1] = in_paramValues1[idx];

	if (in_paramValues2 != nullptr)
		localParam[in_mode2] = in_paramValues2[idx];

	if (in_paramValues3 != nullptr)
		localParam[in_mode3] = in_paramValues3[idx];

	//double localH1 = in_h * localParam[0];
	//double localH2 = in_h * (1 - localParam[0]);

	//Skip PREPEAKFINDER points
	for (size_t i = 0; i < amountOfSkipPoints; ++i) {

		calculateDiscreteModel(in_discreteModelMode, x, localParam, in_h);

		if (resultMode == KDE_MODE && abs(x[in_nValue]) > thresholdValueOfMaxSignalValue)
		{
			in_dataSizes[idx] = 0;
			delete[] localParam;
			return;
		}

		if (resultMode == PEAKFINDER_MODE && abs(x[in_nValue]) > thresholdValueOfMaxSignalValue)
		{
			in_dataSizes[idx] = 0;
			delete[] localParam;
			return;
		}

	}

	//Calculating
	for (size_t i = 0; i < amountOfTPoints; ++i)
	{
		in_data[index + i] = (float)(x[in_nValue]);
		for (size_t j = 0; j < in_prescaller - 1; ++j)
			calculateDiscreteModel(in_discreteModelMode, x, localParam, in_h);
		calculateDiscreteModel(in_discreteModelMode, x, localParam, in_h);

		if (resultMode == KDE_MODE && abs(x[in_nValue]) > thresholdValueOfMaxSignalValue)
		{
			in_dataSizes[idx] = 0;
			delete[] localParam;
			return;
		}

		if (resultMode == PEAKFINDER_MODE && abs(x[in_nValue]) > thresholdValueOfMaxSignalValue)
		{
			in_dataSizes[idx] = 0;
			delete[] localParam;
			return;
		}
	}

	delete[] localParam;

	// Here is the choice of method: KDE or peakFinder
	// TODO add switch on method of result
	// WARNING!!! THIS METHOD MUST TO FILL dataSizes[] (IF NEEDED)!!!

	int outSize = 0;

	switch (resultMode)
	{
	case PEAKFINDER_MODE:
		//		peakFinder(idx, 0, amountOfTPoints, in_data, in_dataSizes, in_data);
		peakFinderForDBSCAN(idx, (float)in_h, (float)0, amountOfTPoints, in_data, in_data, in_dataSizes);
		break;
	case KDE_MODE:
		//outSize = peakFinder(idx, 0, amountOfTPoints, in_data, in_dataSizes, in_data);
		//in_dataSizes[idx] = outSize;
		//kdeMethod(idx, in_data, in_dataSizes, in_kdeSampling, outSize, in_kdeSamplesInterval1, in_kdeSamplesInterval2, amountOfTPoints, in_kdeSmoothH, amountOfTPoints*0.1);


		outSize = peakFinderForDBSCAN(idx, in_h, 0, amountOfTPoints, in_data, in_data, in_dataSizes);

		//float maxx = -9999;//0.002
		//float maxy = -9999; //0.1
		//float minx = 9999;//0.002
		//float miny = 9999; //0.1

		//for (int i = 0; i < outSize; i++) {
		//	if (in_data[index + i * 2] > maxx) {
		//		maxx = in_data[index + i * 2];
		//	}
		//	if (in_data[index + i * 2 + 1] > maxy) {
		//		maxy = in_data[index + i * 2];
		//	}
		//	if (in_data[index + i * 2] < minx) {
		//		minx = in_data[index + i * 2];
		//	}
		//	if (in_data[index + i * 2 + 1] < miny) {
		//		miny = in_data[index + i * 2];
		//	}
		//}


		//maxx = 1 / (maxx - minx);
		//maxy = 1 / (maxy - miny);
		////maxx = 0;
		////maxy = 0;
		////float maxx = 1;
		////float maxy = 0;
		//for (int i = 0; i < outSize; i++) {
		//	in_data[index + i * 2] = (in_data[index + i * 2]) * maxx;
		//	in_data[index + i * 2 + 1] = in_data[index + i * 2 + 1] * maxy;
		//}

		//float maxx = 2;//0.002
		//float maxy = 40; //0.1
		//float minx = -2;//0.002
		//float miny = 0; //0.1
		//maxx = 1 / (maxx - minx);
		//maxy = 1 / (maxy - miny);
		//for (int i = 0; i < outSize; i++) {
		//	in_data[index + i * 2] = (in_data[index + i * 2]) * maxx;
		//	in_data[index + i * 2 + 1] = in_data[index + i * 2 + 1] * maxy;
		//}

		//float maxx = 1;//0.002
		//float maxy = 0; //0.1
		//for (int i = 0; i < outSize; i++) {
		//	in_data[index + i * 2] = (in_data[index + i * 2]) * maxx;
		//	in_data[index + i * 2 + 1] = in_data[index + i * 2 + 1] * maxy;
		//}


		float maxx = 11;//0.002
		float maxy = 8; //0.1
		float minx = -1.5;//0.002
		float miny = 0; //0.1
		float deltx = 1 / (maxx - minx);
		float delty = 1 / (maxy - miny);
		for (int i = 0; i < outSize; i++) {
			in_data[index + i * 2] = (in_data[index + i * 2] - minx) * deltx;
			in_data[index + i * 2 + 1] = (in_data[index + i * 2 + 1] - miny) * delty;
		}

		dbscan(in_data, amountOfTPoints, outSize, idx, 0.01f, in_dataSizes, 0.2 * amountOfTPoints);
//		in_dataSizes[idx] = (int)(miny*1000);
		break;
	}
}


__device__ void calculateDiscreteModel(int mode, double* X, double* a, double h)
{


	switch (mode)
	{
	case ROSSLER: // 0.2 0.2 5.7
		//x[0] = x[0] + localH1 * (-x[1] - x[2]);
		//x[1] = (x[1] + localH1 * (x[0])) / (1 - values[1] * localH1);
		//x[2] = (x[2] + localH1 * values[2]) / (1 - localH1 * (x[0] - values[3]));
		//x[2] = x[2] + localH2 * (values[2] + x[2] * (x[0] - values[3]));
		//x[1] = x[1] + localH2 * (x[0] + values[1] * x[1]);
		//x[0] = x[0] + localH2 * (-x[1] - x[2]);
		break;
	case CHEN: // 40 3 28
		//x[0] = (x[0] + localH1 * values[1] * x[1]) / (1 + localH1 * values[1]);
		//x[1] = (x[1] + localH1 * x[0] * (values[3] - values[1] - x[2])) / (1 - localH1 * values[3]);
		//x[2] = (x[2] + localH1 * x[0] * x[1]) / (1 + localH1 * values[2]);
		//x[2] = x[2] + localH2 * (x[0] * x[1] - values[2] * x[2]);
		//x[1] = x[1] + localH2 * (x[0] * (values[3] - values[1] - x[2]) + values[3] * x[1]);
		//x[0] = x[0] + localH2 * (values[1] * (x[1] - x[0]));
		break;
	case LORENZ: // 10 28 2.6667
		//x[0] = (x[0] + localH1 * values[1] * x[1]) / (1 + localH1 * values[1]);
		//x[1] = (x[1] + localH1 * x[0] * (values[2] - x[2])) / (1 + localH1);
		//x[2] = (x[2] + localH1 * x[0] * x[1]) / (1 + localH1 * values[3]);
		//x[2] = x[2] + localH2 * (x[0] * x[1] - values[3] * x[2]);
		//x[1] = x[1] + localH2 * (x[0] * (values[2] - x[2]) - x[1]);
		//x[0] = x[0] + localH2 * (values[1] * (x[1] - x[0])); 
		break;
	case LORENZ_RYBIN: // -2,5 10 20 3 -0,695
		//x[0] = x[0] + localH1 * (values[2] * x[1] - values[1] * x[0] + values[5] * x[1] * x[2]);
		//x[1] = x[1] + localH1 * (values[3] * x[0] - x[0] * x[2] - x[1]);
		//x[2] = x[2] + localH1 * (x[0] * x[1] - values[4] * x[2]);
		//x[2] = (x[2] + localH2 * (x[0] * x[1])) / (1 + values[4] * localH2);
		//x[1] = (x[1] + localH2 * (values[3] * x[0] - x[0] * x[2])) / (1 + localH2);
		//x[0] = (x[0] + localH2 * (values[2] * x[1] + values[5] * x[1] * x[2])) / (1 + localH2 * values[1]);
		break;
	case CONSERVA: // -2,5 10 20 3 -0,695
		//double h_local = h * 1.35120719196;
		//double h1 = h_local * a[0];
		//double h2 = h_local * (1 - a[0]);4

		//double h_local = h * 0.5;
		//double h1 = h_local * 1.35120719196;
		//double h2 = h1;
		//X[0] = X[0] + h1 * (X[1] + X[0] * X[2]);
		//X[1] = X[1] + h1 * (-a[2] * X[0] + X[1] * X[2] + X[3]);
		//X[2] = X[2] + h1 * (1 - X[0] * X[0] - X[1] * X[1]);
		//X[3] = X[3] + h1 * (-a[1] * X[1]);
		//X[3] = X[3] + h2 * (-a[1] * X[1]);
		//X[2] = X[2] + h2 * (1 - X[0] * X[0] - X[1] * X[1]);
		//X[1] = (X[1] + h2 * (-a[2] * X[0] + X[3])) / (1 - h2 * X[2]);
		//X[0] = (X[0] + h2 * (X[1])) / (1 - h2 * X[2]);
		////h_local = h * (-1.702414383919);
		//h1 = h_local * (-1.702414383919);
		//h2 = h1;
		//X[0] = X[0] + h1 * (X[1] + X[0] * X[2]);
		//X[1] = X[1] + h1 * (-a[2] * X[0] + X[1] * X[2] + X[3]);
		//X[2] = X[2] + h1 * (1 - X[0] * X[0] - X[1] * X[1]);
		//X[3] = X[3] + h1 * (-a[1] * X[1]);
		//X[3] = X[3] + h2 * (-a[1] * X[1]);
		//X[2] = X[2] + h2 * (1 - X[0] * X[0] - X[1] * X[1]);
		//X[1] = (X[1] + h2 * (-a[2] * X[0] + X[3])) / (1 - h2 * X[2]);
		//X[0] = (X[0] + h2 * (X[1])) / (1 - h2 * X[2]);
		//h1 = h_local * 1.35120719196;
		//h2 = h1;
		//X[0] = X[0] + h1 * (X[1] + X[0] * X[2]);
		//X[1] = X[1] + h1 * (-a[2] * X[0] + X[1] * X[2] + X[3]);
		//X[2] = X[2] + h1 * (1 - X[0] * X[0] - X[1] * X[1]);
		//X[3] = X[3] + h1 * (-a[1] * X[1]);
		//X[3] = X[3] + h2 * (-a[1] * X[1]);
		//X[2] = X[2] + h2 * (1 - X[0] * X[0] - X[1] * X[1]);
		//X[1] = (X[1] + h2 * (-a[2] * X[0] + X[3])) / (1 - h2 * X[2]);
		//X[0] = (X[0] + h2 * (X[1])) / (1 - h2 * X[2]);

		//double X1[4];
		//double k[4][4];
		//int N = 4;
		//int i, j;
		//for (i = 0; i < N; i++) {
		//	X1[i] = X[i];
		//}
		//for (j = 0; j < 4; j++) {
		//	k[0][j] = (X1[1] + X1[0] * X1[2]);
		//	k[1][j] = (-a[2] * X1[0] + X1[1] * X1[2] + X1[3]);
		//	k[2][j] = (1 - X1[0] * X1[0] - X1[1] * X1[1]);
		//	k[3][j] = (-a[1] * X1[1]);
		//	if (j == 3) {
		//		for (i = 0; i < N; i++) {
		//			X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
		//		}
		//	}
		//	else if (j == 2) {
		//		for (i = 0; i < N; i++) {
		//			X1[i] = X[i] + h * k[i][j];
		//		}
		//	}
		//	else {
		//		for (i = 0; i < N; i++) {
		//			X1[i] = X[i] + 0.5 * h * k[i][j];
		//		}
		//	}
		//}

		//int M = 4;
		//int N = 4;
		//double X1[4];
		//double k[4][4];
		//double A[4][4];
		//double B[4];
		//int N = 4;
		//int i, j, l;
		//for (i = 0; i < N; i++) {
		//	for (j = 0; i <= i - 1 ; i++) {
		//		for (l = 0; i < M; i++) {
		//			k[l][i] = k[l][i] + k[l][j] * A[i][j];
		//		}
		//	}
		//	for (l = 0; i < M; i++) {
		//		k[l][i] = k[l][i]*h + X[l];
		//	}
		//	k[0][i] = (k[1][i] + k[0][i] * k[2][i]);
		//	k[1][i] = (-a[2] * k[0][i] + k[1][i] * k[2][i] + k[3][i]);
		//	k[2][j] = (1 - k[0][i] * k[0][i] - k[1][i] * k[1][i]);
		//	k[3][j] = (-a[1] * k[1][i]);
		//}
		//for (i = 0; i < N; i++) {
		//	for (l = 0; i < M; i++) {
		//		X[l] = X[l] + h * B[i] * k[l][i];
		//	}
		//}


		break;
	case DISSIPATA:
		////jOSEPHSON jUNCTION double params[6]{ 2.7,0.75,1.2,6.9,0.367,0.0478 };
		//double X1[3];
		//h = h * 0.5;
		//X[0] = X[0] + h * (X[1]);
		//X[1] = X[1] + h * ((1 / a[1]) * (a[2] - ((X[1] > a[3]) ? a[4] : a[5]) * X[1] - sinf( (float)X[0]) - X[2]));
		//X[2] = X[2] + h * ((1 / a[0]) * (X[1] - X[2]));
		//X1[0] = X[0];
		//X1[1] = X[1];
		//X1[2] = X[2];
		//X[2] = (X1[2] + h * (1 / a[0]) * X[1]) / (1 + h * (1 / a[0]));
		//X[1] = X1[1] + h * ((1 / a[1]) * (a[2] - ((X[1] > a[3]) ? a[4] : a[5]) * X[1] - sinf((float)X[0]) - X[2]));
		//X[1] = X1[1] + h * ((1 / a[1]) * (a[2] - ((X[1] > a[3]) ? a[4] : a[5]) * X[1] - sinf((float)X[0]) - X[2]));
		//X[0] = X1[0] + h * (X[1]);

		//Rossler double params[6]{ 0.5,0.2,0.2,5.7 };
		//double X1[3];
		//h = 0.5 * h;
		//X1[0] = X[0] + h * (-X[1] - X[2]);
		//X1[1] = X[1] + h * (X[0] + a[1] * X[1]);
		//X1[2] = X[2] + h * (a[2] + X[2] * (X[0] - a[3]));
		//h = h * 2;
		//X[0] = X[0] + h * (-X1[1] - X1[2]);
		//X[1] = X[1] + h * (X1[0] + a[1] * X1[1]);
		//X[2] = X[2] + h * (a[2] + X1[2] * (X1[0] - a[3]));


		h = 0.5 * h;
		X[0] = X[0] + h * (-X[1] - X[2]);
		X[1] = (X[1] + h * (X[0])) / (1 - a[1] * h);
		X[2] = (X[2] + h * a[2]) / (1 - h * (X[0] - a[3]));
		X[2] = X[2] + h * (a[2] + X[2] * (X[0] - a[3]));
		X[1] = X[1] + h * (X[0] + a[1] * X[1]);
		X[0] = X[0] + h * (-X[1] - X[2]);


		/*
		double w[3][4];
		double h2 = h * 0.5;

		w[0][0] = 1 + a[1] * h2;
		w[0][1] = -h2 * a[1];
		w[0][2] = 0;
		w[1][0] = -h2 * (a[2] - X[2]);
		w[1][1] = 1 + 1 * h2;
		w[1][2] = h2 * X[0];
		w[2][0] = -h2 * X[1];
		w[2][1] = -h2 * X[0];
		w[2][2] = 1 + a[3] * h2;

		w[0][3] = h * a[1] * (X[1] - X[0]);
		w[1][3] = h * (a[2] * X[0] - X[1] - X[0] * X[2]);
		w[2][3] = h * (X[0] * X[1] - a[3] * X[2]);

		int HEIGHT = 3;
		int WIDTH = 4;
		int k; int i; int j; float t; float d;

		for (k = 0; k <= HEIGHT - 2; k++) {

			int l = k;

			for (i = k + 1; i <= HEIGHT - 1; i++) {
				if (abs(w[i][k]) > abs(w[l][k])) {
					l = i;
				}
			}
			if (l != k) {
				for (j = 0; j <= WIDTH - 1; j++) {
					if ((j == 0) || (j >= k)) {
						t = w[k][j];
						w[k][j] = w[l][j];
						w[l][j] = t;
					}
				}
			}

			d = 1.0 / w[k][k];
			for (i = (k + 1); i <= (HEIGHT - 1); i++) {
				if (w[i][k] == 0) {
					continue;
				}
				t = w[i][k] * d;
				for (j = k; j <= (WIDTH - 1); j++) {
					if (w[k][j] != 0) {
						w[i][j] = w[i][j] - t * w[k][j];
					}
				}
			}
		}

		for (i = (HEIGHT); i >= 2; i--) {
			for (j = 1; j <= i - 1; j++) {
				t = w[i - j - 1][i - 1] / w[i - 1][i - 1];
				w[i - j - 1][WIDTH - 1] = w[i - j - 1][WIDTH - 1] - t * w[i - 1][WIDTH - 1];
			}
			w[i - 1][WIDTH - 1] = w[i - 1][WIDTH - 1] / w[i - 1][i - 1];
		}
		w[0][WIDTH - 1] = w[0][WIDTH - 1] / w[0][0];

		X[0] = X[0] + w[0][WIDTH - 1];
		X[1] = X[1] + w[1][WIDTH - 1];
		X[2] = X[2] + w[2][WIDTH - 1];
		*/

		//double X1[3];
		//h = 0.5 * h;
		//X1[0] = X[0] + h * a[1] * (X[1] - X[0]);
		//X1[1] = X[1] + h * (X[0] * (a[2] - X[2]) - X[1]);
		//X1[2] = X[2] + h * (X[0] * X[1] - a[3] * X[2]);
		//h = h * 2;
		//X[0] = X[0] + h * a[1] * (X1[1] - X1[0]);
		//X[1] = X[1] + h * (X1[0] * (a[2] - X1[2]) - X1[1]);
		//X[2] = X[2] + h * (X1[0] * X1[1] - a[3] * X1[2]);

		////Lorenz
		//h = 0.5 * h;
		//X[0] = (X[0] + h * a[1] * X[1]) / (1 + h * a[1]);
		//X[1] = (X[1] + h * X[0] * (a[2] - X[2])) / (1 + h);
		//X[2] = (X[2] + h * X[0] * X[1]) / (1 + h * a[3]);
		//X[2] = X[2] + h * (X[0] * X[1] - a[3] * X[2]);
		//X[1] = X[1] + h * (X[0] * (a[2] - X[2]) - X[1]);
		//X[0] = X[0] + h * (a[1] * (X[1] - X[0]));


		//double h_local = h * 1.35120719196;
		//double h_local = h;
		//double h1 = h_local * a[0];
		//double h2 = h_local * (1 - a[0]);
		////Lorenz
		//X[0] = (X[0] + h1 * a[1] * X[1]) / (1 + h1 * a[1]);
		//X[1] = (X[1] + h1 * X[0] * (a[2] - X[2])) / (1 + h1);
		//X[2] = (X[2] + h1 * X[0] * X[1]) / (1 + h1 * a[3]);
		//X[2] = X[2] + h2 * (X[0] * X[1] - a[3] * X[2]);
		//X[1] = X[1] + h2 * (X[0] * (a[2] - X[2]) - X[1]);
		//X[0] = X[0] + h2 * (a[1] * (X[1] - X[0]));
		//h_local = h * (-1.702414383919);
		//h1 = h_local * a[0];
		//h2 = h_local * (1 - a[0]);
		//X[0] = (X[0] + h1 * a[1] * X[1]) / (1 + h1 * a[1]);
		//X[1] = (X[1] + h1 * X[0] * (a[2] - X[2])) / (1 + h1);
		//X[2] = (X[2] + h1 * X[0] * X[1]) / (1 + h1 * a[3]);
		//X[2] = X[2] + h2 * (X[0] * X[1] - a[3] * X[2]);
		//X[1] = X[1] + h2 * (X[0] * (a[2] - X[2]) - X[1]);
		//X[0] = X[0] + h2 * (a[1] * (X[1] - X[0]));
		//h_local = h * 1.35120719196;
		//h1 = h_local * a[0];
		//h2 = h_local * (1 - a[0]);
		//X[0] = (X[0] + h1 * a[1] * X[1]) / (1 + h1 * a[1]);
		//X[1] = (X[1] + h1 * X[0] * (a[2] - X[2])) / (1 + h1);
		//X[2] = (X[2] + h1 * X[0] * X[1]) / (1 + h1 * a[3]);
		//X[2] = X[2] + h2 * (X[0] * X[1] - a[3] * X[2]);
		//X[1] = X[1] + h2 * (X[0] * (a[2] - X[2]) - X[1]);
		//X[0] = X[0] + h2 * (a[1] * (X[1] - X[0]));


		/*double X1[3];
		double k[3][4];
		int N = 4;
		int i, j;
		for (i = 0; i < N; i++) {
			X1[i] = X[i];
		}
		for (j = 0; j < 4; j++) {
		k[0][j] = (a[1] * (X1[1] - X1[0]));
		k[1][j] = (X1[0] * (a[2] - X1[2]) - X1[1]);
		k[2][j] = (X1[0] * X1[1] - a[3] * X1[2]);
			if (j == 3) {
				for (i = 0; i < N; i++) {
					X1[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
				}
			}
			else if (j == 2) {
				for (i = 0; i < N; i++) {
					X1[i] = X[i] + h * k[i][j];
				}
			}
			else {
				for (i = 0; i < N; i++) {
					X1[i] = X[i] + 0.5 * h * k[i][j];
				}
			}
		}
		for (i = 0; i < N; i++) {
			X[i] = X1[i];
		}*/

		break;
	case TIMUR:
		//double X1[3];
		//double k[3][4];
		//int N = 3;
		//int i, j;
		//for (i = 0; i < N; i++) {
		//	X1[i] = X[i];
		//}
		//for (j = 0; j < 4; j++) {
		//	k[0][j] = (X1[2] - X1[0] * (a[1] + a[2])) * a[3] * a[4];
		//	k[1][j] = (a[5] * X1[0] * X1[2] / 10 * a[6] * a[7] - X1[1] * (a[8] + a[2])) * a[3] * a[4];
		//	k[2][j] = (-a[5] * X1[0] * a[9] + a[5] * X1[1] * a[10]) * a[11];

		//	if (j == 3) {
		//		for (i = 0; i < N; i++) {
		//			X1[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
		//		}
		//	}
		//	else if (j == 2) {
		//		for (i = 0; i < N; i++) {
		//			X1[i] = X[i] + h * k[i][j];
		//		}
		//	}
		//	else {
		//		for (i = 0; i < N; i++) {
		//			X1[i] = X[i] + 0.5 * h * k[i][j];
		//		}
		//	}
		//}
		//for (i = 0; i < N; i++) {
		//	X[i] = X1[i];
		//}
		//break;
	case CompCD:

		double h_local = h * 0.5;
		double h1 = h_local * 1.35120719196;
		double h2 = h1;
		X[0] = X[0] + h1 * (X[1] + X[0] * X[2]);
		X[1] = X[1] + h1 * (-a[2] * X[0] + X[1] * X[2] + X[3]);
		X[2] = X[2] + h1 * (1 - X[0] * X[0] - X[1] * X[1]);
		X[3] = X[3] + h1 * (-a[1] * X[1]);
		X[3] = X[3] + h2 * (-a[1] * X[1]);
		X[2] = X[2] + h2 * (1 - X[0] * X[0] - X[1] * X[1]);
		X[1] = (X[1] + h2 * (-a[2] * X[0] + X[3])) / (1 - h2 * X[2]);
		X[0] = (X[0] + h2 * (X[1])) / (1 - h2 * X[2]);
		//h_local = h * (-1.702414383919);
		h1 = h_local * (-1.702414383919);
		h2 = h1;
		X[0] = X[0] + h1 * (X[1] + X[0] * X[2]);
		X[1] = X[1] + h1 * (-a[2] * X[0] + X[1] * X[2] + X[3]);
		X[2] = X[2] + h1 * (1 - X[0] * X[0] - X[1] * X[1]);
		X[3] = X[3] + h1 * (-a[1] * X[1]);
		X[3] = X[3] + h2 * (-a[1] * X[1]);
		X[2] = X[2] + h2 * (1 - X[0] * X[0] - X[1] * X[1]);
		X[1] = (X[1] + h2 * (-a[2] * X[0] + X[3])) / (1 - h2 * X[2]);
		X[0] = (X[0] + h2 * (X[1])) / (1 - h2 * X[2]);
		h1 = h_local * 1.35120719196;
		h2 = h1;
		X[0] = X[0] + h1 * (X[1] + X[0] * X[2]);
		X[1] = X[1] + h1 * (-a[2] * X[0] + X[1] * X[2] + X[3]);
		X[2] = X[2] + h1 * (1 - X[0] * X[0] - X[1] * X[1]);
		X[3] = X[3] + h1 * (-a[1] * X[1]);
		X[3] = X[3] + h2 * (-a[1] * X[1]);
		X[2] = X[2] + h2 * (1 - X[0] * X[0] - X[1] * X[1]);
		X[1] = (X[1] + h2 * (-a[2] * X[0] + X[3])) / (1 - h2 * X[2]);
		X[0] = (X[0] + h2 * (X[1])) / (1 - h2 * X[2]);
		break;
	case RK4:
		double X1[4];
		double k[4][4];
		int N = 4;
		int i, j;
		for (i = 0; i < N; i++) {
			X1[i] = X[i];
		}
		for (j = 0; j < 4; j++) {
			k[0][j] = (X1[1] + X1[0] * X1[2]);
			k[1][j] = (-a[2] * X1[0] + X1[1] * X1[2] + X1[3]);
			k[2][j] = (1 - X1[0] * X1[0] - X1[1] * X1[1]);
			k[3][j] = (-a[1] * X1[1]);
			if (j == 3) {
				for (i = 0; i < N; i++) {
					X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
				}
			}
			else if (j == 2) {
				for (i = 0; i < N; i++) {
					X1[i] = X[i] + h * k[i][j];
				}
			}
			else {
				for (i = 0; i < N; i++) {
					X1[i] = X[i] + 0.5 * h * k[i][j];
				}
			}
		}
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

__device__ int peakFinderForDBSCAN(int idx, float in_h, float prePeakFinder, size_t amountOfTPoints, float* in_data, float* out_data, int* out_dataSizes)
{
	size_t index = idx * amountOfTPoints;
	int _outSize = 0;

	for (int i = 3 + prePeakFinder * amountOfTPoints; i < amountOfTPoints - 1; ++i)
	{
		if (in_data[index + i] > in_data[index + i - 1] && in_data[index + i] > in_data[index + i + 1])
		{
			out_data[index + _outSize * 2] = in_data[index + i];
			out_data[index + _outSize * 2 + 1] = i;
			++_outSize;
		}
		else if (in_data[index + i] > in_data[index + i - 1] && in_data[index + i] == in_data[index + i + 1])
		{
			for (size_t k = i; k < amountOfTPoints - 1; ++k)
			{
				if (in_data[index + k] < in_data[index + k + 1])
				{
					break;
					i = k;
				}
				if (in_data[index + k] == in_data[index + k + 1])
					continue;
				if (in_data[index + k] > in_data[index + k + 1])
				{
					out_data[index + _outSize * 2] = in_data[index + k];
					out_data[index + _outSize * 2 + 1] = k;
					_outSize++;
					i = k + 1;
					break;
				}
			}
		}
	}

	if (_outSize > 1) {
		for (size_t i = 0; i < _outSize - 1; i++)
		{
			out_data[index + i * 2] = out_data[index + i * 2 + 2];
			out_data[index + i * 2 + 1] = (float)((out_data[index + i * 2 + 3] - out_data[index + i * 2 + 1]) * in_h);
		}
		_outSize = _outSize * 1 - 1;
	}
	else {
		_outSize = 0;
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
	float kdeSmoothH,
	int criticalValueOfPeaks)
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

	if (_outSize > criticalValueOfPeaks)
	{
		kdeResult[idx] = criticalValueOfPeaks;
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
			double tempData = (data[idx * amountOfTPoints + m] - delt) / kdeSmoothH;
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
__host__ void linspace(T1 a, T1 b, int amount, T2 * out, int startIndex)
{
	if (amount <= 0)
		throw std::invalid_argument("linspace error. amount <= 0");
	if (amount == 1)
	{
		out[0] = a;
		return;
	}

	double step = (b - a) / (amount - 1);
	for (size_t i = 0; i < amount; ++i)
		out[startIndex + i] = a + i * step;

	return;
}

__device__ float customAbs(float value)
{
	if (value < 0)
		return -value;
	return value;
}


__device__ float distance(float x1, float y1, float x2, float y2)
{
	if (x1 == x2 && y1 == y2)
		return 0;
	float dx = x2 - x1;
	float dy = y2 - y1;

	return hypotf(dx, dy);
}


__device__ void expand_cluster(float* input, int index, int amountOfPeaks, int p, float eps)
{
	for (int i = index + 0; i < index + (amountOfPeaks * 2 - 2); i += 2) {
		if (distance(input[i + 1], input[i + 3], input[p + 1], input[p + 3]) < eps) {
			float temp = input[i + 1];
			input[i + 1] = temp > 0 ? -temp : temp;

			if (i != p)
				expand_cluster(input, index, amountOfPeaks, i, eps);
		}
	}
}


__device__ int dbscan(float* input, int amountOfTPoints, int amountOfPeaks, int idx, float eps, int* dataSizes, int criticalValueOfPeaks)
{
	if (amountOfPeaks <= 0)
	{
		dataSizes[idx] = 0;
		return;
	}

	if (amountOfPeaks == 1)
	{
		dataSizes[idx] = 1;
		return;
	}


	if (amountOfPeaks > criticalValueOfPeaks)
	{
		dataSizes[idx] = 0;
		return;
	}


	//dbscan(in_data, amountOfTPoints, outSize, idx, 0.5f, in_dataSizes);
	int cluster = 0;
	int NumNeibor = 0;


	int index = amountOfTPoints * idx;

	for (int i = index + amountOfPeaks * 2; i < index + amountOfPeaks * 4; i++) {
		input[i] = 0;
	}

	for (int i = 0; i < amountOfPeaks; i++)
		if (NumNeibor >= 1)
		{
			i = input[index + amountOfPeaks * 3 + NumNeibor - 1];
			input[index + amountOfPeaks * 3 + NumNeibor - 1] = 0;
			NumNeibor = NumNeibor - 1;
			for (int k = 0; k < amountOfPeaks - 1; k++) {
				if (i != k && input[index + amountOfPeaks * 2 + k] == 0) {
					//if (distance(input[index + i], input[index + i + 1], input[index + k], input[index + k + 1])<= eps) {
					if (distance(input[index + i * 2], input[index + i * 2 + 1], input[index + k * 2], input[index + k * 2 + 1]) < eps) {
						input[index + amountOfPeaks * 2 + k] = cluster;
						input[index + amountOfPeaks * 3 + k] = k;
						NumNeibor++;
					}
				}
			}
		}
		else if (input[index + amountOfPeaks * 2 + i] == 0) {
			NumNeibor = 0;
			cluster++;
			input[index + amountOfPeaks * 2 + i] = cluster;
			for (int k = 0; k < amountOfPeaks - 1; k++) {
				if (i != k && input[index + amountOfPeaks * 2 + k] == 0) {
					//if (distance(input[index + i], input[index + i + 1], input[index + k], input[index + k + 1])<= eps) {
					if (distance(input[index + i * 2], input[index + i * 2 + 1], input[index + k * 2], input[index + k * 2 + 1]) < eps) {
						input[index + amountOfPeaks * 2 + k] = cluster;
						input[index + amountOfPeaks * 3 + k] = k;
						NumNeibor++;
					}
				}
			}
		}
	//for (int i = 0; i < amountOfPeaks; i++) {

	//}
	//for (int i = index + amountOfPeaks * 2; i < index + amountOfPeaks * 4; i++) {
	//	input[i] = 0;
	//}

	dataSizes[idx] = cluster;

	return;// cluster - 1;
}


__host__ void getParamsAndSymmetry2D(double* param1, double* param2,
	double startInterval1, double finishInteraval1,
	double startInterval2, double finishInteraval2,
	int nPts)
{
	double* tempParams = new double[nPts];
	linspace(startInterval2, finishInteraval2, nPts, tempParams);

	for (int i = 0; i < nPts; ++i)
	{
		linspace(startInterval1, finishInteraval1, nPts, param1, i * nPts);
		for (int j = 0; j < nPts; ++j)
			param2[nPts * i + j] = tempParams[i];
	}

	delete[] tempParams;
}


__host__ void getParamsAndSymmetry3D(double* param1, double* param2, double* param3,
	double startInterval1, double finishInteraval1,
	double startInterval2, double finishInteraval2,
	double startInterval3, double finishInteraval3,
	int nPts)
{
	{
		double* tempParams2 = new double[nPts];
		double* tempParams3 = new double[nPts];

		linspace(startInterval2, finishInteraval2, nPts, tempParams2);
		linspace(startInterval3, finishInteraval3, nPts, tempParams3);

		for (int k = 0; k < nPts; ++k)
			for (int i = 0; i < nPts; ++i)
			{
				linspace(startInterval1, finishInteraval1, nPts, param1, i * nPts + k * nPts * nPts);
				for (int j = 0; j < nPts; ++j)
				{
					param2[nPts * nPts * k + nPts * i + j] = tempParams2[i];
					param3[nPts * nPts * k + nPts * i + j] = tempParams3[k];
				}
			}

		delete[] tempParams2;
		delete[] tempParams3;
	}
}


template <class T>
__host__ void slice(T * in, int a, int b, T * out)
{
	if (b - a < 0)
		throw std::invalid_argument("slice error. b < a");
	for (size_t i = 0; i < b - a; ++i)
		out[i] = in[a + i];
}


