#pragma once

#include "Object.h"

template <class T = double> 
class Cube : public Object<T>
{
public:
	T width, height, depth;

	Cube() = default;
	Cube(vec3<T> origin, vec3<T> rotation, T width, T height, T depth, cv::Vec<T,4> color) : width(width), height(height), depth(depth), Object<T>(origin, rotation, color) {};
	~Cube() = default;

	T CheckCollision(RayLight<T> ray, vec3<T>& collision, vec3<T>& normal);
	void CheckCollisionCuda(std::vector<T>& out, CudaPointers<T>& cp);
};