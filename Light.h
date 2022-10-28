#pragma once

#include "RayLight.h"
#include "CudaPointers.h"

#include <opencv2/opencv.hpp>

template<class T>
class Light
{
public:
	vec3<T> color;

	Light(vec3<T> color) : color(color) {};

	static void setAmbientLightCUDA(CudaPointers<T>& cp, T ambientValue, int count);

	virtual void addLightEffectsCUDA(CudaPointers<T>& cp, vec3<T>& viewerPos, int count) {};
};

