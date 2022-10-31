#pragma once

#include "RayLight.h"
#include "CudaPointers.h"
#include "Object.h"

#include <opencv2/opencv.hpp>

template<class T>
class Light
{
public:
	vec3<T> color;

	Light(vec3<T> color) : color(color) {};

	static cv::Vec<T, 4> setAmbientLight(Object<T>& obj, T ambientValue);
	static void setAmbientLightCUDA(CudaPointers<T>& cp, T ambientValue, int count);

	virtual cv::Vec<T, 4> lightEffect(Object<T>& obj, vec3<T>& collision, vec3<T>& normal, vec3<T>& viewerPos) { return cv::Vec<T, 4>(0, 0, 0, 0); }
	virtual void addLightEffectsCUDA(CudaPointers<T>& cp, int count) {};
};

