#pragma once

#include <math.h>

template <class T>
struct vec3 {
	T x, y, z;

	vec3(T x, T y, T z) : x(x), y(y), z(z) {};

	void normalize() {
		T norm = x * x + y * y + z * z;
		if (norm == 1.) return;
		norm = 1./sqrt(norm);
		x *= norm;
		y *= norm;
		z *= norm;
	}

	inline vec3 add(T x2, T y2, T z2) {
		return vec3(x + x2, y + y2, z + z2);
	}

	inline vec3 operator+(vec3 n) {
		return vec3(x + n.x, y + n.y, z + n.z);
	}

	inline vec3 operator-(vec3 n) {
		return vec3(x - n.x, y - n.y, z - n.z);
	}

	inline vec3 operator*(T value) {
		return vec3(value * x, value * y, value * z);
	}

	inline T dot(vec3<T> v) {
		return x * v.x + y * v. y + z * v.z;
	}
};

template <class T = double>
class RayLight
{
public:
	vec3<T> origin, direction;

	RayLight() : origin(vec3<T>(0, 0, 0)), direction(vec3<T>(0, 0, 0)) {};
	RayLight(vec3<T> origin, vec3<T> direction) : origin(origin), direction(direction) {};
	RayLight(T ox, T oy, T oz, T dx, T dy, T dz) : origin(vec3<T>(ox, oy, oz)), direction(vec3<T>(dx, dy, dz)) {};

	inline vec3<T> addOrigin(T ox, T oy, T oz) {
		return origin + vec3<T>(ox,oy,oz);
	}

	inline vec3<T> scaleDirection(T value) {
		return value*direction;
	}
};