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

	T CheckCollision(RayLight<T> ray) {
		T radius = diameter / 2.;
		
		ray.direction.normalize();

		vec3<T> dirD = this->rotation * degToRad;

		T cx = cos(dirD.x), sx = sin(dirD.x), cy = cos(dirD.y), sy = sin(dirD.y), cz = cos(dirD.z), sz = sin(dirD.z);

		vec3<T> newOrigin = vec3<T>(
			cz * cy * ray.origin.x + sz * cy * ray.origin.y - sy * ray.origin.z,
			(cz * sy * sx - sz * cx) * ray.origin.x + (sz * sy * sx + cz * cx) * ray.origin.y + (cy * sx) * ray.origin.z,
			(cz * sy * cx + sz * sx) * ray.origin.x + (sz * sy * cx - cz * sx) * ray.origin.y + (cy * cx) * ray.origin.z);

		vec3<T> newDirection = vec3<T>(
			(cz * cy) * ray.direction.x + sz * cy * ray.direction.y - sy * ray.direction.z,
			(cz * sy * sx - sz * cx) * ray.direction.x + (sz * sy * sx + cz * cx) * ray.direction.y + (cy * sx) * ray.direction.z,
			(cz * sy * cx + sz * sx) * ray.direction.x + (sz * sy * cx - cz * sx) * ray.direction.y + cy * cx * ray.direction.z);

		vec3<T> relOrigin = newOrigin - this->position;

		T a = newDirection.x * newDirection.x + newDirection.z * newDirection.z;
		T b = 2.*(newDirection.x * relOrigin.x + newDirection.z * relOrigin.z);
		T c = relOrigin.x * relOrigin.x + relOrigin.z * relOrigin.z - radius * radius;

		T delta = -4. * a * c + b * b;
		if (delta < 0) goto topBottomCollision;
		{
			T dist = (-b - sqrt(delta)) / (2. * a);
			T yCol = relOrigin.y + dist * newDirection.y;

			if (yCol <= height / 2. && yCol >= -height / 2.) return dist;
		}
	topBottomCollision:
		{
			T yTarget = height / 2.;
			T dist = (yTarget - relOrigin.y) / newDirection.y;
			T distAux = (-yTarget - relOrigin.y) / newDirection.y;
			if (distAux >= 0 && distAux < dist) dist = distAux;
			if (dist >= 0) {
				T xCol = relOrigin.x + dist * newDirection.x, zCol = relOrigin.z + dist * newDirection.z;
				if (xCol*xCol+zCol*zCol <= radius*radius)
					return dist;
			}
		}
		return -1;
	}
};