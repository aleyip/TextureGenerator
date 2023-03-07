#include "Object.h"
#include "defines.h"

template <class T>
T Object<T>::CheckCollision(RayLight<T> ray, vec3<T> &collision, vec3<T> &normal) {
	collision = vec3<T>(0, 0, 0);
	normal = vec3<T>(0, 0, 0);
	return 10e10;
}

template <class T>
void Object<T>::CheckCollisionCuda(CudaPointers<T>& cp, int count, int8_t objindex) {
}

template class Object<typeT>;