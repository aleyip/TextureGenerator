#pragma once
#include "Object.h"
#include "RayLight.h"
#include "CudaPointers.h"

template <class T=double> class Sphere : public Object<T>
{
public:
	T diameter;
	
	Sphere() = default;
	Sphere(vec3<T> origin, T diameter, cv::Vec<T, 3> color, T specular = .3, uint8_t specularShininness = 32) : diameter(diameter), Object<T>(origin, vec3<T>(0,0,0), color, specular, specularShininness) {};
	~Sphere() = default;

	T CheckCollision(RayLight<T> ray, vec3<T>& collision, vec3<T>& normal);
	void CheckCollisionCuda(CudaPointers<T>& cp, int count, int8_t objindex);
	void Report();
};