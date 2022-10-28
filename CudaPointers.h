#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

#include "RayLight.h"

template<class T>
class CudaPointers
{
public:
	T* d_rayList = 0, * d_dist = 0, * d_position = 0, * d_rotation = 0;
	T* d_collision = 0, * d_normal = 0;
	T* d_objcolor = 0;
	T* d_color = 0;
	int8_t* d_hitobject = 0;
	uint8_t* d_objShin = 0;
	int* d_size = 0;

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

	void downloadPixelColor(cv::Vec<T, 4>* data, int count);
	
	void allocate(int totalSize, int usize, int vsize, int ssize, int tsize);
	void free();
};

