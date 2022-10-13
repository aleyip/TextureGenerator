#pragma once

#include "RayLight.h"

template<class T>
class CudaPointers
{
public:
	T* d_rayList = 0, * d_out = 0, * d_position = 0;
	int* d_size = 0;

	CudaPointers() = default;
	~CudaPointers() {
		free();
	}
	
	void allocate(int totalSize, int usize, int vsize, int ssize, int tsize);
	void free();
};

