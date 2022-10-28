#pragma once

#include "Object.h"

template <class T> 
class Cube : public Object<T>
{
public:
	T width, height, depth;

	Cube() = default;
	Cube(vec3<T> origin, vec3<T> rotation, T width, T height, T depth, cv::Vec<T, 3> color, T specular = .3, uint8_t specularShininness = 32) : width(width), height(height), depth(depth), Object<T>(origin, rotation, color, specular, specularShininness) {};
	~Cube() = default;

	T CheckCollision(RayLight<T> ray, vec3<T>& collision, vec3<T>& normal);
	void CheckCollisionCuda(CudaPointers<T>& cp, int count, int8_t objindex);
};