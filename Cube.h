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

	T CheckCollision(RayLight<T> ray) {
		T mod = ray.direction.dot(ray.direction);
		if (mod != 1.) {
			mod = sqrt(mod);
			ray.direction = ray.direction * (1. / mod);
		}

		vec3<T> dirD = this->rotation * degToRad;

		T cx = cos(dirD.x), sx = sin(dirD.x), cy = cos(dirD.y), sy = sin(dirD.y), cz = cos(dirD.z), sz = sin(dirD.z);

		vec3<T> newOrigin = vec3<T>(
			cz * cy * ray.origin.x + sz * cy * ray.origin.y - sy * ray.origin.z,
			(cz * sy * sx - sz * cx) * ray.origin.x + (sz * sy * sx + cz * cx) * ray.origin.y + (cy * sx) * ray.origin.z,
			(cz * sy * cx + sz * sx) * ray.origin.x + (sz * sy * cx - cz * sx) * ray.origin.y+ (cy * cx) * ray.origin.z);

		vec3<T>  newDirection = vec3<T>(
			cz * cy * ray.direction.x + sz * cy * ray.direction.y - sy * ray.direction.z,
			(cz * sy * sx - sz * cx) * ray.direction.x + (sz * sy * sx + cz * cx) * ray.direction.y + (cy * sx) * ray.direction.z,
			(cz * sy * cx + sz * sx) * ray.direction.x + (sz * sy * cx - cz * sx) * ray.direction.y + (cy * cx) * ray.direction.z);

		vec3<T>  relOrigin = newOrigin - this->position;

		T dist = -1;
		{
			T xTarget = depth / 2.;
			dist = (xTarget - relOrigin.x) / newDirection.x;
			T distAux = (-xTarget - relOrigin.x) / newDirection.x;
			if (distAux >= 0 && distAux < dist) dist = distAux;
			if (dist >= 0) {
				T yCol = relOrigin.y + dist * newDirection.y, zCol = relOrigin.z + dist * newDirection.z;
				if (yCol >= -height / 2 && yCol <= height / 2 &&
					zCol >= -width / 2 && zCol <= width / 2)
					return dist;
			}
		}
		{
			T yTarget = height / 2.;
			dist = (yTarget - relOrigin.y) / newDirection.y;
			T distAux = (-yTarget - relOrigin.y) / newDirection.y;
			if (distAux >= 0 && distAux < dist) dist = distAux;
			if (dist >= 0) {
				T xCol = relOrigin.x + dist * newDirection.x, zCol = relOrigin.z + dist * newDirection.z;
				if (xCol >= -depth / 2 && xCol <= depth / 2 &&
					zCol >= -width / 2 && zCol <= width / 2)
					return dist;
			}
		}
		{
			T zTarget = width / 2.;
			dist = (zTarget - relOrigin.z) / newDirection.z;
			T distAux = (zTarget - relOrigin.z) / newDirection.z;
			if (distAux >= 0 && distAux < dist) dist = distAux;
			if (dist >= 0) {
				T xCol = relOrigin.x + dist * newDirection.x, yCol = relOrigin.y + dist * newDirection.y;
				if (xCol >= -depth / 2 && xCol <= depth / 2 &&
					yCol >= -height / 2 && yCol <= height / 2)
					return dist;
			}
		}

		return  -1;
	}
};