#pragma once
#define M_PI 	3.14159265358979323846

#include "Object.h"

template <class T = double> class Cylinder : public Object<T>
{
public:
	T diameter, height;

	Cylinder() = default;
	Cylinder(vec3<T> origin, vec3<T> rotation, T diameter, T height, cv::Vec<T, 3> color, T specular = .3, uint8_t specularShininness = 32) : diameter(diameter), height(height), Object<T>(origin, rotation, color, specular, specularShininness) {};
	~Cylinder() = default;

	T CheckCollision(RayLight<T> ray, vec3<T>& collision, vec3<T>& normal);
	void CheckCollisionCuda(CudaPointers<T>& cp, int count, int8_t objindex);
	void Report();
};