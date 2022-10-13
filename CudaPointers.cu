#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaPointers.h"

template<class T>
void CudaPointers<T>::allocate(int totalSize, int usize, int vsize, int ssize, int tsize) {
	cudaError_t cudaStatus;

	int size[] = { usize, vsize, ssize, tsize };

	// Allocate device memory
	cudaStatus = cudaMalloc((void**)&d_rayList, sizeof(RayLight<T>) * totalSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_rayList!\n");
		goto ErrorAllocate;
	}

	cudaStatus = cudaMalloc((void**)&d_position, sizeof(vec3<T>));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_position!\n");
		goto ErrorAllocate;
	}

	cudaStatus = cudaMalloc((void**)&d_out, sizeof(T) * totalSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_out!\n");
		goto ErrorAllocate;
	}

	cudaStatus = cudaMalloc((void**)&d_size, 4 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_size!\n");
		goto ErrorAllocate;
	}

	// Transfer data from host to device memory
	cudaStatus = cudaMemcpy(d_size, size, 4 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_size Host to Device!\n");
		goto ErrorAllocate;
	}

ErrorAllocate:
	return;
}

template<class T>
void CudaPointers<T>::uploadRayList(std::vector<RayLight<T>> rayList) {
	cudaError_t cudaStatus;
	// Transfer data from host to device memory
	cudaStatus = cudaMemcpy(d_rayList, rayList.data(), sizeof(RayLight<T>) * rayList.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_rayList Host to Device!\n");
	}
}

template<class T>
void CudaPointers<T>::downloadRayList(std::vector<RayLight<T>>& rayList) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(rayList.data(), d_rayList, sizeof(RayLight<T>) * rayList.size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_rayList Device to Host!\n");
	}
}

template<class T>
void CudaPointers<T>::free() {
	cudaFree(d_rayList);
	cudaFree(d_out);
	cudaFree(d_position);
	cudaFree(d_size);
}

template class CudaPointers<double>;
template class CudaPointers<float>;