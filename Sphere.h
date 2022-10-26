#pragma once
#include "Object.h"
#include "RayLight.h"
#include "CudaPointers.h"

template <class T=double> class Sphere : public Object<T>
{
public:
	T diameter;
	
	Sphere() = default;
	Sphere(vec3<T> origin, T diameter, cv::Vec<T,4> color) : diameter(diameter), Object<T>(origin, vec3<T>(0,0,0), color) {};
	~Sphere() = default;

	T CheckCollision(RayLight<T> ray, vec3<T>& collision, vec3<T>& normal);
	void CheckCollisionCuda(std::vector<T>& out, CudaPointers<T>& cp);
};