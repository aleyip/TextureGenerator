#include <stdio.h>
#include "defines.h"

#include <opencv2/opencv.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

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

	cudaStatus = cudaMalloc((void**)&d_collision, sizeof(vec3<T>) * totalSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_collision!\n");
		goto ErrorAllocate;
	}

	cudaStatus = cudaMalloc((void**)&d_normal, sizeof(vec3<T>) * totalSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_normal!\n");
		goto ErrorAllocate;
	}

	cudaStatus = cudaMalloc((void**)&d_dist, sizeof(T) * totalSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_out!\n");
		goto ErrorAllocate;
	}

	cudaStatus = cudaMalloc((void**)&d_hitobject, sizeof(int8_t) * totalSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_hitobject!\n");
		goto ErrorAllocate;
	}

	cudaStatus = cudaMalloc((void**)&d_position, sizeof(vec3<T>));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_position!\n");
		goto ErrorAllocate;
	}

	cudaStatus = cudaMalloc((void**)&d_rotation, sizeof(vec3<T>));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_rotation!\n");
		goto ErrorAllocate;
	}

	cudaStatus = cudaMalloc((void**)&d_color, sizeof(vec3<T>));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_lightcolor!\n");
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
void CudaPointers<T>::uploadNormalList(std::vector<vec3<T>> normalList) {
	cudaError_t cudaStatus;
	// Transfer data from host to device memory
	cudaStatus = cudaMemcpy(d_normal, normalList.data(), sizeof(vec3<T>) * normalList.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_normal Host to Device!\n");
	}
}

template<class T>
void CudaPointers<T>::downloadNormalList(std::vector<vec3<T>>& normalList) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(normalList.data(), d_normal, sizeof(vec3<T>) * normalList.size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_normal Device to Host!\n");
	}
}

template<class T>
void CudaPointers<T>::uploadCollisionList(std::vector<vec3<T>> collisionList) {
	cudaError_t cudaStatus;
	// Transfer data from host to device memory
	cudaStatus = cudaMemcpy(d_collision, collisionList.data(), sizeof(vec3<T>) * collisionList.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_collision Host to Device!\n");
	}
}

template<class T>
void CudaPointers<T>::downloadCollisionList(std::vector<vec3<T>>& collisionList) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(collisionList.data(), d_collision, sizeof(vec3<T>) * collisionList.size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_collision Device to Host!\n");
	}
}

template<class T>
void CudaPointers<T>::uploadDistList(std::vector<T> distList) {
	cudaError_t cudaStatus;
	// Transfer data from host to device memory
	cudaStatus = cudaMemcpy(d_dist, distList.data(), sizeof(T) * distList.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_dist Host to Device!\n");
	}
}

template<class T>
void CudaPointers<T>::downloadDistList(std::vector<T>& distList) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(distList.data(), d_dist, sizeof(T) * distList.size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_dist Device to Host!\n");
	}
}

template<class T>
void CudaPointers<T>::setDistList(T value, int sizeList) {
	thrust::device_ptr<T> dev_ptr(d_dist);
	thrust::fill(dev_ptr, dev_ptr+sizeList, value);
}

template<class T>
void CudaPointers<T>::uploadHitObjectList(std::vector<int8_t> hitobjectList) {
	cudaError_t cudaStatus;
	// Transfer data from host to device memory
	cudaStatus = cudaMemcpy(d_hitobject, hitobjectList.data(), sizeof(int8_t) * hitobjectList.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_dist Host to Device!\n");
	}
}

template<class T>
void CudaPointers<T>::downloadHitObjectList(std::vector<int8_t>& hitobjectList) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(hitobjectList.data(), d_hitobject, sizeof(int8_t) * hitobjectList.size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_hitobject Device to Host!\n");
	}
}

template<class T>
void CudaPointers<T>::setHitObjectList(int8_t value, int sizeList) {
	thrust::device_ptr<int8_t> dev_ptr(d_hitobject);
	thrust::fill(dev_ptr, dev_ptr + sizeList, value);
}

template<class T>
void CudaPointers<T>::uploadObjectColorProp(std::vector<cv::Vec<T, 4>> colorList, std::vector<uint8_t> shinyList) {
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&d_objcolor, colorList.size() * sizeof(cv::Vec<T, 4>));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_objcolor!\n");
		return;
	}

	cudaStatus = cudaMalloc((void**)&d_objShin, shinyList.size() * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_objShin!\n");
		return;
	}

	cudaStatus = cudaMemcpy(d_objcolor, colorList.data(), sizeof(cv::Vec<T, 4>) * colorList.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_objcolor Host to Device!\n");
	}

	cudaStatus = cudaMemcpy(d_objShin, shinyList.data(), sizeof(uint8_t) * shinyList.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_objShin Host to Device!\n");
	}
}

template<class T>
void CudaPointers<T>::downloadPixelColor(cv::Vec<T, 4>* data, int count) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(data, d_rayList, sizeof(cv::Vec<T, 4>) * count, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_rayList Device to Host as PixelColor!\n");
	}
}

template<class T>
void CudaPointers<T>::free() {
	cudaFree(d_rayList);
	cudaFree(d_collision);
	cudaFree(d_normal);
	cudaFree(d_dist);
	cudaFree(d_hitobject);
	cudaFree(d_position);
	cudaFree(d_rotation);
	cudaFree(d_color);
	cudaFree(d_objcolor);
	cudaFree(d_size);
}

template class CudaPointers<typeT>;