#include "Object.h"

template <class T>
T Object<T>::CheckCollision(RayLight<T> ray) {
	return -1;
}

template <class T>
void Object<T>::CheckCollisionCuda(std::vector<T> &out, CudaPointers<T>& cp) {
	for (T& value : out)
		value = -1;
}

template class Object<double>;
template class Object<float>;