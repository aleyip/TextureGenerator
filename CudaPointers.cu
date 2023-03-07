#include <stdio.h>
#include "defines.h"

#include <opencv2/opencv.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "CudaPointers.h"

template<class T, int n>
__global__ void DataReduction_kernel(T* out, const int wsTotal, const int pixelnum){
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < pixelnum) {
		T* pixel = out + n * index;
		T sumPixel[n];
		sumPixel[0] = 0;
#pragma unroll
		for (int i = 0; i < n; i++)
			sumPixel[i] = 0;
		for (int i = 0; i < wsTotal; i++)
		{
#pragma unroll
			for (int i = 0; i < n; i++)
				sumPixel[i] += pixel[i];
			pixel += n * pixelnum;
		}
		pixel = out + n * index;
#pragma unroll
		for(int i = 0; i<n; i++)
			pixel[i] = sumPixel[i] / wsTotal;
	}
}

template<class T>
__global__ void VectorReduction_kernel(T* out, const int wsTotal, const int pixelnum) {
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < pixelnum) {
		T* pixel = out + 3 * index;
		T sumPixel[] = { 0,0,0 };
		for (int i = 0; i < wsTotal; i++)
		{
			sumPixel[0] += pixel[0];
			sumPixel[1] += pixel[1];
			sumPixel[2] += pixel[2];
			pixel += 3 * pixelnum;
		}
		pixel = out + 3 * index;

		T mod = sumPixel[0] * sumPixel[0] + sumPixel[1] * sumPixel[1] + sumPixel[2] * sumPixel[2];
		mod = 1 / sqrt(mod);

		pixel[0] = sumPixel[0] *= mod;
		pixel[1] = sumPixel[1] *= mod;
		pixel[2] = sumPixel[2] *= mod;
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
void CudaPointers<T>::uploadRayList(RayLight<T>* rayList, int length) {
	cudaError_t cudaStatus;
	// Transfer data from host to device memory
	cudaStatus = cudaMemcpy(d_rayList, rayList, sizeof(RayLight<T>) * length, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_rayList Host to Device!\n");
	}
}

template<class T>
void CudaPointers<T>::downloadRayList(RayLight<T>* rayList, int length) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(rayList, d_rayList, sizeof(RayLight<T>) * length, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_rayList Device to Host!\n");
	}
}

template <class T>
cudaError_t NormalReduction_wrapper(int length, int wsTotal, CudaPointers<T>& cp) {
	cudaError_t cudaStatus;

	// Executing kernel 
	{
		dim3 threadsPerBlock(1024);
		dim3 blocksPerGrid(ceil(double(length) / double(threadsPerBlock.x)));
		VectorReduction_kernel<T> << < blocksPerGrid, threadsPerBlock >> > (cp.d_normalList, wsTotal, length);
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Normal Reduction kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto ErrorReduction;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Normal Reduction Kernel!\n", cudaStatus);
		goto ErrorReduction;
	}
ErrorReduction:
	return cudaStatus;
}

template<class T>
void CudaPointers<T>::normalReduction(int wsTotal, int length) {
	cudaError_t cudaStatus = NormalReduction_wrapper<T>(length, wsTotal, *this);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Normal Reduction Failed\n");
	}
}

template<class T>
void CudaPointers<T>::uploadNormalList(vec3<T>* normalList, int length) {
	cudaError_t cudaStatus;
	// Transfer data from host to device memory
	cudaStatus = cudaMemcpy(d_normalList, normalList, sizeof(vec3<T>) * length, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_normal Host to Device!\n");
	}
}

template<class T>
void CudaPointers<T>::downloadNormalList(vec3<T>* normalList, int length) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(normalList, d_normalList, sizeof(vec3<T>) * length, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_normal Device to Host!\n");
	}
}

template<class T>
void CudaPointers<T>::uploadCollisionList(vec3<T>* collisionList, int length) {
	cudaError_t cudaStatus;
	// Transfer data from host to device memory
	cudaStatus = cudaMemcpy(d_collisionList, collisionList, sizeof(vec3<T>) * length, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_collision Host to Device!\n");
	}
}

template<class T>
void CudaPointers<T>::downloadCollisionList(vec3<T>* collisionList, int length) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(collisionList, d_collisionList, sizeof(vec3<T>) * length , cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_collision Device to Host!\n");
	}
}

template <class T>
cudaError_t DistReduction_wrapper(int length, int wsTotal, CudaPointers<T>& cp) {
	cudaError_t cudaStatus;

	// Executing kernel 
	{
		dim3 threadsPerBlock(1024);
		dim3 blocksPerGrid(ceil(double(length) / double(threadsPerBlock.x)));
		DataReduction_kernel<T, 1> << < blocksPerGrid, threadsPerBlock >> > (cp.d_distList, wsTotal, length);
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Dist Reduction kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto ErrorReduction;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Dist Reduction Kernel!\n", cudaStatus);
		goto ErrorReduction;
	}
ErrorReduction:
	return cudaStatus;
}

template<class T>
void CudaPointers<T>::distReduction(int wsTotal, int length) {
	cudaError_t cudaStatus = DistReduction_wrapper<T>(length, wsTotal, *this);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Dist Reduction Failed\n");
	}
}

template<class T>
void CudaPointers<T>::uploadDistList(T* distList, int length) {
	cudaError_t cudaStatus;
	// Transfer data from host to device memory
	cudaStatus = cudaMemcpy(d_distList, distList, sizeof(T) * length, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_dist Host to Device!\n");
	}
}

template<class T>
void CudaPointers<T>::downloadDistList(T* distList, int length) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(distList, d_distList, sizeof(T) * length, cudaMemcpyDeviceToHost);
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
void CudaPointers<T>::uploadHitObjectList(int8_t *hitobjectList, int length) {
	cudaError_t cudaStatus;
	// Transfer data from host to device memory
	cudaStatus = cudaMemcpy(d_hitobjectList, hitobjectList, sizeof(int8_t) * length, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_dist Host to Device!\n");
	}
}

template<class T>
void CudaPointers<T>::downloadHitObjectList(int8_t* hitobjectList, int length) {
	cudaError_t cudaStatus;
	// Transfer data back to host memory
	cudaStatus = cudaMemcpy(hitobjectList, d_hitobjectList, sizeof(int8_t) * length, cudaMemcpyDeviceToHost);
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
void CudaPointers<T>::uploadObjectColorProp(cv::Vec<T, 4>* colorList, uint8_t* shinyList, int length) {
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&d_objcolorList, sizeof(cv::Vec<T, 4>) * length);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_objcolor!\n");
		return;
	}

	cudaStatus = cudaMalloc((void**)&d_objShinList, sizeof(uint8_t) * length);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in d_objShin!\n");
		return;
	}

	cudaStatus = cudaMemcpy(d_objcolorList, colorList, sizeof(cv::Vec<T, 4>) * length, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in d_objcolor Host to Device!\n");
	}

	cudaStatus = cudaMemcpy(d_objShinList, shinyList, sizeof(uint8_t) * length, cudaMemcpyHostToDevice);
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
		DataReduction_kernel<T,4> << < blocksPerGrid, threadsPerBlock >> > (cp.d_colorList, wsTotal, length);
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