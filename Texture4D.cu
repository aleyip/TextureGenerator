#include "Texture4D.h"
#include "Object.h"

#include <stdio.h>
#include <iostream>

#include <chrono>

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

const char lookupTable[] = "0123456789abcdef";

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ inline void atomicAdd(double* address, double value) {
	unsigned long long oldval, newval, readback;
	oldval = __double_as_longlong(*address);
	newval = __double_as_longlong(__longlong_as_double(oldval) + value);
	while ((readback = atomicCAS((unsigned long long*) address, oldval, newval)) != oldval) {
		oldval = readback;
		newval = __double_as_longlong(__longlong_as_double(oldval) + value);
	}
}
#endif

template<class T>
__device__ void normalize(T* vec) {
	T norm = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
	norm = sqrt(norm);
	vec[0] /= norm;
	vec[1] /= norm;
	vec[2] /= norm;
}

template<class T>
__global__ void raylightgenerator_kernel(T* out, int v, int s, int t, int* size, T radius, int ws, int wsTotal) {
	int u = blockIdx.x;
	int k = threadIdx.x;
	int l = threadIdx.y;
	int m = threadIdx.z;
	int n = blockIdx.y;

	T angle[] = { M_PI * (2 * ((n - ws) / (T)(ws - 1) - u) / size[0] + 1),
		M_PI * ((v + (m - ws) / (T)(ws - 1)) / (size[1] - 1) - 0.5),
		M_PI * ((s + (l - ws) / (T)(ws - 1)) / (size[2] - 1) - 0.5),
		M_PI * ((t + (k - ws) / (T)(ws - 1)) / (size[3] - 1) - 0.5), };

	T pos[3] = { radius * sin(angle[0]) * cos(angle[1]), radius * sin(angle[1]), radius * cos(angle[0]) * cos(angle[1]) };
	T dir[3] = { sin(angle[2]) * cos(angle[3]), sin(angle[3]), cos(angle[2]) * cos(angle[3]) };

	normalize<T>(pos);
	T versorRight[] = { -pos[2], 0., pos[0] };
	normalize<T>(versorRight);
	T versorUp[] = { -versorRight[2] * pos[1], versorRight[2] * pos[0] - versorRight[0] * pos[2], versorRight[0] * pos[1] };
	normalize<T>(versorUp);

	T vD[] = {
		-(versorRight[0] * dir[0] + versorUp[0] * dir[1] + pos[0] * dir[2]),
		-(versorRight[1] * dir[0] + versorUp[1] * dir[1] + pos[1] * dir[2]),
		-(versorRight[2] * dir[0] + versorUp[2] * dir[1] + pos[2] * dir[2])
	};

	int index = 6 * (u * wsTotal + ws * (ws * (ws * n + m) + l) + k);
	out[index] = pos[0];
	out[index + 1] = pos[1];
	out[index + 2] = pos[2];
	out[index + 3] = dir[0];
	out[index + 4] = dir[1];
	out[index + 5] = dir[2];
}

template<class T>
cudaError_t RayLightGenerator_wrapper(std::vector<RayLight<T>>& out, int v, int s, int t, int* size, T radius, int ws, int wsTotal, CudaPointers<T>& cp) {
	cudaError_t cudaStatus;

	//auto start = std::chrono::steady_clock::now();
	// Executing kernel 
	{
		dim3 threadsPerBlock(ws, ws, ws);
		dim3 blocksPerGrid(size[0], ws);
		//std::cout << threadsPerBlock << " " << blocksPerGrid << std::endl;
		raylightgenerator_kernel<T> << < blocksPerGrid, threadsPerBlock >> > (cp.d_rayList, v, s, t, cp.d_size, radius, ws, wsTotal);
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Ray Light kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto ErrorRayLight;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Ray Light Kernel!\n", cudaStatus);
		goto ErrorRayLight;
	}

	//auto end = std::chrono::steady_clock::now();
	//std::cout << "Cuda Light Generator Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

	// Transfer data back to host memory
	//cudaStatus = cudaMemcpy(out.data(), cp.d_rayList, sizeof(RayLight<T>) * size[0] * wsTotal, cudaMemcpyDeviceToHost);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed in d_out Device to Host!\n");
	//	goto ErrorRayLight;
	//}

	//for (int i = 0; i < 100; i++)
	//{
	//	printf("%03d \t %f %f %f \t %f %f %f\n", i, out[i].origin.x, out[i].origin.y, out[i].origin.z, out[i].direction.x, out[i].direction.y, out[i].direction.z);
	//}

ErrorRayLight:
	return cudaStatus;
}

inline void print_hex(char* out, const unsigned char value)
{
	out[0] = lookupTable[value >> 4];
	out[1] = lookupTable[value & 0xf];
}

template <class T>
void Texture4D<T>::compileToUnity(std::string s) {
	std::ofstream fout;
	std::string name;
	size_t indexfirst = s.find_last_of('\\') + 1;
	size_t indexlast = s.find_last_of('.');
	name = s.substr(indexfirst, indexlast - indexfirst);

	std::cout << "Gerando arquivo: " << name << std::endl;

	fout.open(s, std::ofstream::binary);
	if (fout.is_open())
	{
		fout << "%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!117 &11700000\n";
		fout << "Texture3D:" << std::endl;
		fout << "  m_ObjectHideFlags: 0" << std::endl;
		fout << "  m_CorrespondingSourceObject: {fileID: 0}" << std::endl;
		fout << "  m_PrefabInstance: {fileID: 0}" << std::endl;
		fout << "  m_PrefabAsset: {fileID: 0}" << std::endl;
		fout << "  m_Name: " << name << std::endl;
		fout << "  m_ImageContentsHash:" << std::endl << "    serializedVersion: 2" << std::endl << "    Hash: 00000000000000000000000000000000" << std::endl;
		fout << "  m_ForcedFallbackFormat: 4" << std::endl;
		fout << "  m_DownscaleFallback: 0" << std::endl;
		fout << "  m_IsAlphaChannelOptional: 0" << std::endl;
		fout << "  serializedVersion: 3" << std::endl;
		fout << "  m_ColorSpace: 0" << std::endl;
		fout << "  m_Format: 8" << std::endl;
		fout << "  m_Width: " << usize << std::endl;
		fout << "  m_Height: " << vsize << std::endl;
		fout << "  m_Depth: " << ssize * tsize << std::endl;
		fout << "  m_MipCount: 1" << std::endl;
		fout << "  m_DataSize: " << 4 * usize * vsize * ssize * tsize << std::endl;
		fout << "  m_TextureSettings:" << std::endl;
		fout << "    serializedVersion: 2" << std::endl;
		fout << "    m_FilterMode: 0" << std::endl;
		fout << "    m_Aniso: 1" << std::endl;
		fout << "    m_MipBias: 0" << std::endl;
		fout << "    m_WrapU: 0" << std::endl;
		fout << "    m_WrapV: 1" << std::endl;
		fout << "    m_WrapW: 1" << std::endl;
		fout << "  m_UsageMode: 0" << std::endl;
		fout << "  m_IsReadable: 1" << std::endl;
		fout << "  image data: " << 4 * usize * vsize * ssize * tsize << std::endl;
		fout << "  _typelessdata: ";
		cv::Mat img;
		texture.convertTo(img, CV_8UC4, 255);

		std::string fileData = std::string(8 * totalSize, 0x00); //2 de 2 char por value e 4 de 4 cores
		for (int t = 0; t < tsize; t++)
			for (int s = 0; s < ssize; s++)
				for (int v = 0; v < vsize; v++)
					for (int u = 0; u < usize; u++)
					{
						//size_t index = 8 * (u + usize * (v + vsize * (s + t * tsize)));
						size_t index = 8 * (u + usize * (v + vsize * (s + t * tsize)));
						cv::Vec4b pixel = img.at<cv::Vec4b>(getCoord(u, v, s, t));
						print_hex((char*)&fileData[index], pixel[2]);
						print_hex((char*)(&fileData[index] + 2), pixel[1]);
						print_hex((char*)(&fileData[index] + 4), pixel[0]);
						print_hex((char*)(&fileData[index] + 6), pixel[3]);
					}
		fout << fileData << std::endl;
		fout << "  m_StreamData:" << std::endl;
		fout << "    serializedVersion: 2 " << std::endl;
		fout << "    offset : 0 " << std::endl;
		fout << "    size: 0 " << std::endl;
		fout << "    path : " << std::endl;

		fout.close();
	}
}

template <class T>
void Texture4D<T>::RayLightGeneratorCuda(std::vector<RayLight<T>>& out, int v, int s, int t, T radius, int ws, int wsTotal, CudaPointers<T> &cp) {
	int size[] = { usize, vsize, ssize, tsize };

	cudaError_t cudaStatus = RayLightGenerator_wrapper<T>(out, v, s, t, size, radius, ws, wsTotal, cp);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Ray Light Generator Failed\n");
	}
}

template class Texture4D<float>;
template class Texture4D<double>;