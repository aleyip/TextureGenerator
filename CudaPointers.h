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

	void uploadRayList(RayLight<T>* rayList, int length);
	void downloadRayList(RayLight<T>* rayList, int length);

	void normalReduction(int wsTotal, int count);
	void uploadNormalList(vec3<T>* normalList, int length);
	void downloadNormalList(vec3<T>* normalList, int length);

	void uploadCollisionList(vec3<T>* collisionList, int length);
	void downloadCollisionList(vec3<T>* collisionList, int length);

	void distReduction(int wsTotal, int count);
	void uploadDistList(T* distList, int length);
	void downloadDistList(T* distList, int length);
	void setDistList(T value, int sizeList);

	void uploadHitObjectList(int8_t* hitobjectList, int length);
	void downloadHitObjectList(int8_t* hitobjectList, int length);
	void setHitObjectList(int8_t value, int sizeList);

	void uploadObjectColorProp(cv::Vec<T, 4>* colorList, uint8_t *shinyList, int length);

	void pixelReduction(int wsTotal, int count);
	void downloadPixelColor(cv::Vec<T, 4>* data, int count, int start = 0);
	
	void allocate(int totalSize, int usize, int vsize, int ssize, int tsize);
	void free();
};

