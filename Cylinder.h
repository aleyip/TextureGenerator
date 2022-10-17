#pragma once
#define M_PI 	3.14159265358979323846

#include "Object.h"

template <class T = double> class Cylinder : public Object<T>
{
public:
	T diameter, height;

	Cylinder() = default;
	Cylinder(vec3<T> origin, vec3<T> rotation, T diameter, T height, cv::Vec<T,4> color) : diameter(diameter), height(height), Object<T>(origin, rotation, color) {};
	~Cylinder() = default;

	T CheckCollision(RayLight<T> ray);
	void CheckCollisionCuda(std::vector<T>& out, CudaPointers<T>& cp);
};