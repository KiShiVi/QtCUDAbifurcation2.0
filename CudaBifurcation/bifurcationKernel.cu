#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "bifurcationKernel.cuh"

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
							float*				in_params,
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

	size_t freeMemory;
	size_t totalMemory;

	cudaMemGetInfo(&freeMemory, &totalMemory);

	freeMemory *= in_memoryLimit * 0.95;

	float maxMemoryLimit = sizeof(float) * ((in_tMax / in_h) + 1 + in_amountOfParams) + sizeof(int);

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

	float	*	d_data;
	int		*	d_dataSizes;
	float	*	d_dataTimes;
	float	*	d_params;
	float	*	d_initialConditions;

	cudaMalloc((void**)&d_params, in_amountOfParams * sizeof(float));
	cudaMalloc((void**)&d_initialConditions, in_amountOfParams * sizeof(float));
	cudaMemcpy(d_params, in_params, in_amountOfParams * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(d_initialConditions, in_initialConditions, in_amountOfParams * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

	size_t amountOfIteration = (size_t)std::ceilf((float)in_nPts / (float)nPtsLimiter);;

	std::ofstream outFileStream;
	outFileStream.open(in_outPath);

	for (size_t i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
		{
			h_dataTimes = (float*)malloc((in_nPts - nPtsLimiter * i) * sizeof(float));

			slice(globalParamValues, nPtsLimiter * i, in_nPts, h_dataTimes);
			nPtsLimiter = in_nPts - (nPtsLimiter * i);
		}
		else
		{
			h_dataTimes = (float*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(float));
			slice(globalParamValues, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_dataTimes);
		}


		h_data = (float*)malloc(nPtsLimiter * amountOfTPoints * sizeof(float));
		h_dataSizes = (int*)malloc(nPtsLimiter * sizeof(int));

		cudaMalloc((void**)&d_data, nPtsLimiter * amountOfTPoints * sizeof(float));
		cudaMalloc((void**)&d_dataSizes, nPtsLimiter * sizeof(int));
		cudaMalloc((void**)&d_dataTimes, nPtsLimiter * sizeof(float));

		cudaMemcpy(d_dataTimes, h_dataTimes, nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

		int blockSize;
		int minGridSize;
		int gridSize;

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bifuractionKernel, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;



		//Call CUDA func
			bifuractionKernel << <gridSize, blockSize >> > (	nPtsLimiter,
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
																d_params,
																d_dataTimes,
																in_mode);


		cudaMemcpy(h_data, d_data, amountOfTPoints * nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaMemcpy(h_dataSizes, d_dataSizes, nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);

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
			std::cout << "       " << std::setprecision(3) << (100.0f / (float)amountOfIteration) * (i + 1) << "%\n";

		progress.store((100.0f / (float)amountOfIteration) * (i + 1), std::memory_order_seq_cst);
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
							std::atomic<int>& progress)
{
	std::ofstream outFileStream;
	outFileStream.open(in_outPath);
	outFileStream << in_paramValues1 << ", " << in_paramValues2 << "\n" << in_paramValues3 << ", " << in_paramValues4 << "\n";

	size_t amountOfTPoints = in_tMax / in_h;

	float* paramValues1 = nullptr;
	float* paramValues2 = nullptr;

	paramValues1 = (float*)malloc(sizeof(float) * in_nPts * in_nPts);
	paramValues2 = (float*)malloc(sizeof(float) * in_nPts * in_nPts);

	getParamsAndSymmetry2D(paramValues1, paramValues2,
		in_paramValues1, in_paramValues2,
		in_paramValues3, in_paramValues4,
		in_nPts);

	size_t freeMemory;
	size_t totalMemory;

	cudaMemGetInfo(&freeMemory, &totalMemory);

	freeMemory *= in_memoryLimit * 0.95;

	float maxMemoryLimit = sizeof(float) * ((in_tMax / in_h) + 2 + in_amountOfParams) + sizeof(int);

	size_t nPtsLimiter = freeMemory / maxMemoryLimit;

	if (nPtsLimiter <= 0)
	{
		if (in_debug)
			std::cout << "\nVery low memory size. Increase the MEMORY_LIMIT!" << "\n";
		exit(1);
	}

	int*   h_kdeResult;
	float* h_data;
	float* h_paramValues1;
	float* h_paramValues2;

	float* d_params;
	int*   d_kdeResult;
	float* d_data;
	float* d_paramValues1;
	float* d_paramValues2;
	float* d_initialConditions;


	cudaMalloc((void**)&d_params, in_amountOfParams * sizeof(float));
	cudaMalloc((void**)&d_initialConditions, in_amountOfParams * sizeof(float));
	cudaMemcpy(d_params, in_params, in_amountOfParams * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(d_initialConditions, in_initialConditions, in_amountOfParams * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

	size_t amountOfIteration = (size_t)std::ceilf((float)(in_nPts * in_nPts) / (float)nPtsLimiter);

	int stringCounter = 0;

	for (size_t i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
		{
			h_kdeResult   = (int*)malloc(((in_nPts * in_nPts) - nPtsLimiter * i) * sizeof(int));
			h_paramValues1 = (float*)malloc(((in_nPts * in_nPts) - nPtsLimiter * i) * sizeof(float));
			h_paramValues2 = (float*)malloc(((in_nPts * in_nPts) - nPtsLimiter * i) * sizeof(float));

			slice(paramValues1, nPtsLimiter * i, (in_nPts * in_nPts), h_paramValues1);
			slice(paramValues2, nPtsLimiter * i, (in_nPts * in_nPts), h_paramValues2);
			nPtsLimiter = (in_nPts * in_nPts) - (nPtsLimiter * i);
		}
		else
		{
			h_kdeResult = (int*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(int));
			h_paramValues1 = (float*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(float));
			h_paramValues2 = (float*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(float));

			slice(paramValues1, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_paramValues1);
			slice(paramValues2, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_paramValues2);
		}


		h_data = (float*)malloc(nPtsLimiter * amountOfTPoints * sizeof(float));

		cudaMalloc((void**)&d_kdeResult, nPtsLimiter * sizeof(int));
		cudaMalloc((void**)&d_data, nPtsLimiter * amountOfTPoints * sizeof(float));
		cudaMalloc((void**)&d_paramValues1, nPtsLimiter * sizeof(float));
		cudaMalloc((void**)&d_paramValues2, nPtsLimiter * sizeof(float));

		cudaMemcpy(d_paramValues1,	h_paramValues1, nPtsLimiter			* sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaMemcpy(d_paramValues2,	h_paramValues2, nPtsLimiter			* sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

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
			d_params,
			d_paramValues1,
			in_mode1,
			d_paramValues2,
			in_mode2,
			in_kdeSampling,
			in_kdeSamplesInterval1,
			in_kdeSamplesInterval2,
			in_kdeSamplesSmooth);


		//cudaMemcpy(h_data, d_data, amountOfTPoints * nPtsLimiter * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
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
			std::cout << "       " << std::setprecision(3) << (100.0f / (float)amountOfIteration) * (i + 1) << "%\n";

		progress.store((100.0f / (float)amountOfIteration) * (i + 1), std::memory_order_seq_cst);
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
	float* in_params,
	float* in_paramValues1,
	int	in_mode1,
	float* in_paramValues2,
	int in_mode2,
	int in_kdeSampling,
	float in_kdeSamplesInterval1,
	float in_kdeSamplesInterval2,
	float in_kdeSmoothH
	)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= in_nPts)
		return;

	size_t amountOfTPoints = in_TMax / in_h;

	float x[3]{ in_initialConditions[0], in_initialConditions[1], in_initialConditions[2] };


	float* localParam = new float[in_amountOfParams];
	for (int i = 0; i < in_amountOfParams; ++i)
		localParam[i] = in_params[i];

	localParam[in_mode1] = in_paramValues1[idx];

	if (in_paramValues2 != nullptr)
		localParam[in_mode2] = in_paramValues2[idx];

	float localH1 = in_h * localParam[0];
	float localH2 = in_h * (1 - localParam[0]);

	for (size_t i = 0; i < amountOfTPoints; ++i)
	{
		in_data[idx * amountOfTPoints + i] = x[in_nValue];

		x[0] = x[0] + localH1 * (-x[1] - x[2]);
		x[1] = (x[1] + localH1 * (x[0])) / (1 - localParam[1] * localH1);
		x[2] = (x[2] + localH1 * localParam[2]) / (1 - localH1 * (x[0] - localParam[3]));

		x[2] = x[2] + localH2 * (localParam[2] + x[2] * (x[0] - localParam[3]));
		x[1] = x[1] + localH2 * (x[0] + localParam[1] * x[1]);
		x[0] = x[0] + localH2 * (-x[1] - x[2]);

		if (resultMode == KDE_MODE && abs(x[in_nValue]) > thresholdValueOfMaxSignalValue)
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
		//in_dataSizes[idx] = outSize;
		kdeMethod(idx, in_data, in_dataSizes, in_kdeSampling, outSize, in_kdeSamplesInterval1, in_kdeSamplesInterval2, amountOfTPoints, in_kdeSmoothH, 10 * in_TMax * (1 - in_prePeakFinderSliceK));
		break;
	}

	delete[] localParam;
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
		kdeResult[idx] = -1;
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



__host__ void getParamsAndSymmetry2D(float* param1, float* param2,
	float startInterval1, float finishInteraval1,
	float startInterval2, float finishInteraval2,
	int nPts)
{
	float* tempParams = new float[nPts];
	linspace(startInterval2, finishInteraval2, nPts, tempParams);

	for (int i = 0; i < nPts; ++i)
	{
		linspace(startInterval1, finishInteraval1, nPts, param1, i * nPts);
		for (int j = 0; j < nPts; ++j)
			param2[nPts * i + j] = tempParams[i];
	}

	delete[] tempParams;
}



template <class T>
__host__ void slice(T* in, int a, int b, T* out)
{
	if (b - a < 0)
		throw std::invalid_argument("slice error. b < a");
	for (size_t i = 0; i < b - a; ++i)
		out[i] = in[a + i];
}


