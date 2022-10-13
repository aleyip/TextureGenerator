#pragma once

#define M_PI 	3.14159265358979323846
#define M_PI_2 	1.57079632679489661923
#define degToRad 0.01745329251994329576923690768489

#include "RayLight.h"
#include "CudaPointers.h"

#include <opencv2/opencv.hpp>
#include <vector>

template <class T = double> 
class Object
{
public:
	vec3<T> position;

	/// <summary>
	/// Vetor contendo os 3 valores de rotacao em graus na ordem x -> y -> z
	/// </summary>
	vec3<T> rotation;

	cv::Vec4d color;

	Object() = default;
	Object(vec3<T> position, vec3<T> rotation, cv::Vec<T,4> color) : position(position), rotation(rotation), color(color) {};
	virtual ~Object() = default;

	virtual T CheckCollision(RayLight<T> ray);
	virtual void CheckCollisionCuda(std::vector<T>& out, CudaPointers<T>& cp);
};

#include "Object.cu"
