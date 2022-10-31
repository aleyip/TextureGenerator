#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

#include "RayLight.h"

template<class T>
class CudaPointers
{
public:
	T* d_rayList = 0; // vec6 : bufferSize
	T* d_collisionList = 0, * d_normalList = 0; // vec3 : bufferSize
	T* d_colorList = 0; // vec4 : bufferSize
	T* d_distList = 0; // value : bufferSize
	int8_t* d_hitobjectList = 0; // char : bufferSize

	T* d_objcolorList = 0; // vec4 : objSize
	uint8_t* d_objShinList = 0; // char : objSize

	T* d_color = 0, * d_position = 0, * d_rotation = 0; // vec3
	int* d_size = 0; // int4

	CudaPointers() = default;
	~CudaPointers() {
		free();
	}

	void uploadRayList(std::vector<RayLight<T>> rayList);
	void downloadRayList(std::vector<RayLight<T>>& rayList);

	void uploadNormalList(std::vector<vec3<T>> normalList);
	void downloadNormalList(std::vector<vec3<T>>& normalList);

	void uploadCollisionList(std::vector<vec3<T>> collisionList);
	void downloadCollisionList(std::vector<vec3<T>>& collisionList);

	void uploadDistList(std::vector<T> distList);
	void downloadDistList(std::vector<T>& distList);
	void setDistList(T value, int sizeList);

	void uploadHitObjectList(std::vector<int8_t> hitobjectList);
	void downloadHitObjectList(std::vector<int8_t>& hitobjectList);
	void setHitObjectList(int8_t value, int sizeList);

	void uploadObjectColorProp(std::vector<cv::Vec<T, 4>> colorList, std::vector<uint8_t> shinyList);

	void pixelReduction(int wsTotal, int count);
	void downloadPixelColor(cv::Vec<T, 4>* data, int count);
	
	void allocate(int totalSize, int usize, int vsize, int ssize, int tsize);
	void free();
};

