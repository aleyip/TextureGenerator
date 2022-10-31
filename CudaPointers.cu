#include <stdio.h>
#include "defines.h"

#include <opencv2/opencv.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "CudaPointers.h"

template<class T>
__global__ void PixelReduction_kernel(T* out, const int wsTotal, const int pixelnum){
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < pixelnum) {
		T* pixel = out + 4 * index;
		T sumPixel[] = { 0,0,0,0 };
		for (int i = 0; i < wsTotal; i++)
		{
			sumPixel[0] += pixel[0];
			sumPixel[1] += pixel[1];
			sumPixel[2] += pixel[2];
			sumPixel[3] += pixel[3];
			pixel += 4 * pixelnum;
		}
		pixel = out + 4 * index;
		pixel[0] = sumPixel[0] / wsTotal;
		pixel[1] = sumPixel[1] / wsTotal;
		pixel[2] = sumPixel[2] / wsTotal;
		pixel[3] = sumPixel[3] / wsTotal;
	}
}

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

	cudaStatus = cudaMalloc((void**)&d_collisionList, sizeof(vec3<T>) * totalSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_collision!\n");
		goto ErrorAllocate;
	}

	cudaStatus = cudaMalloc((void**)&d_normalList, sizeof(vec3<T>) * totalSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_normal!\n");
		goto ErrorAllocate;
	}

	cudaStatus = cudaMalloc((void**)&d_colorList, sizeof(cv::Vec<T,4>) * totalSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_normal!\n");
		goto ErrorAllocate;
	}

	cudaStatus = cudaMalloc((void**)&d_distList, sizeof(T) * totalSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_out!\n");
		goto ErrorAllocate;
	}

	cudaStatus = cudaMalloc((void**)&d_hitobjectList, sizeof(int8_t) * totalSize);
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
	cudaStatus = cudaMemcpy(d_normalList, normalList.data(), sizeof(vec3<T>) * normalList.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_normal Host to Device!\n");
	}
}

template<class T>
void CudaPointers<T>::downloadNormalList(std::vector<vec3<T>>& normalList) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(normalList.data(), d_normalList, sizeof(vec3<T>) * normalList.size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_normal Device to Host!\n");
	}
}

template<class T>
void CudaPointers<T>::uploadCollisionList(std::vector<vec3<T>> collisionList) {
	cudaError_t cudaStatus;
	// Transfer data from host to device memory
	cudaStatus = cudaMemcpy(d_collisionList, collisionList.data(), sizeof(vec3<T>) * collisionList.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_collision Host to Device!\n");
	}
}

template<class T>
void CudaPointers<T>::downloadCollisionList(std::vector<vec3<T>>& collisionList) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(collisionList.data(), d_collisionList, sizeof(vec3<T>) * collisionList.size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_collision Device to Host!\n");
	}
}

template<class T>
void CudaPointers<T>::uploadDistList(std::vector<T> distList) {
	cudaError_t cudaStatus;
	// Transfer data from host to device memory
	cudaStatus = cudaMemcpy(d_distList, distList.data(), sizeof(T) * distList.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_dist Host to Device!\n");
	}
}

template<class T>
void CudaPointers<T>::downloadDistList(std::vector<T>& distList) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(distList.data(), d_distList, sizeof(T) * distList.size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_dist Device to Host!\n");
	}
}

template<class T>
void CudaPointers<T>::setDistList(T value, int sizeList) {
	thrust::device_ptr<T> dev_ptr(d_distList);
	thrust::fill(dev_ptr, dev_ptr+sizeList, value);
}

template<class T>
void CudaPointers<T>::uploadHitObjectList(std::vector<int8_t> hitobjectList) {
	cudaError_t cudaStatus;
	// Transfer data from host to device memory
	cudaStatus = cudaMemcpy(d_hitobjectList, hitobjectList.data(), sizeof(int8_t) * hitobjectList.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_dist Host to Device!\n");
	}
}

template<class T>
void CudaPointers<T>::downloadHitObjectList(std::vector<int8_t>& hitobjectList) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(hitobjectList.data(), d_hitobjectList, sizeof(int8_t) * hitobjectList.size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_hitobject Device to Host!\n");
	}
}

template<class T>
void CudaPointers<T>::setHitObjectList(int8_t value, int sizeList) {
	thrust::device_ptr<int8_t> dev_ptr(d_hitobjectList);
	thrust::fill(dev_ptr, dev_ptr + sizeList, value);
}

template<class T>
void CudaPointers<T>::uploadObjectColorProp(std::vector<cv::Vec<T, 4>> colorList, std::vector<uint8_t> shinyList) {
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&d_objcolorList, colorList.size() * sizeof(cv::Vec<T, 4>));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_objcolor!\n");
		return;
	}

	cudaStatus = cudaMalloc((void**)&d_objShinList, shinyList.size() * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_objShin!\n");
		return;
	}

	cudaStatus = cudaMemcpy(d_objcolorList, colorList.data(), sizeof(cv::Vec<T, 4>) * colorList.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_objcolor Host to Device!\n");
	}

	cudaStatus = cudaMemcpy(d_objShinList, shinyList.data(), sizeof(uint8_t) * shinyList.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_objShin Host to Device!\n");
	}
}

template <class T>
cudaError_t PixelReduction_wrapper(int length, int wsTotal, CudaPointers<T>& cp) {
	cudaError_t cudaStatus;

	// Executing kernel 
	{
		dim3 threadsPerBlock(1024);
		dim3 blocksPerGrid(ceil(double(length) / double(threadsPerBlock.x)));
		PixelReduction_kernel<T> << < blocksPerGrid, threadsPerBlock >> > (cp.d_colorList, wsTotal, length);
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Pixel Reduction kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto ErrorReduction;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Pixel Reduction Kernel!\n", cudaStatus);
		goto ErrorReduction;
	}
ErrorReduction:
	return cudaStatus;
}

template<class T>
void CudaPointers<T>::pixelReduction(int wsTotal, int length) {
	cudaError_t cudaStatus = PixelReduction_wrapper<T>(length, wsTotal, *this);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Pixel Reduction Failed\n");
	}
}

template<class T>
void CudaPointers<T>::downloadPixelColor(cv::Vec<T, 4>* data, int count) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(data, d_colorList, sizeof(cv::Vec<T, 4>) * count, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_rayList Device to Host as PixelColor!\n");
	}
}

template<class T>
void CudaPointers<T>::free() {
	cudaFree(d_rayList);
	cudaFree(d_collisionList);
	cudaFree(d_normalList);
	cudaFree(d_distList);
	cudaFree(d_colorList);
	cudaFree(d_objcolorList);
	cudaFree(d_hitobjectList);
	cudaFree(d_objShinList);
	cudaFree(d_position);
	cudaFree(d_rotation);
	cudaFree(d_color);
	cudaFree(d_size);
}

template class CudaPointers<typeT>;