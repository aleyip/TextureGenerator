#include "Object.h"
#include "defines.h"

template <class T>
T Object<T>::CheckCollision(RayLight<T> ray, vec3<T> &collision, vec3<T> &normal) {
	collision = vec3<T>(0, 0, 0);
	normal = vec3<T>(0, 0, 0);
	return -1;
}

template <class T>
void Object<T>::CheckCollisionCuda(CudaPointers<T>& cp, int count, int8_t objindex) {
}

template <class T>
T Object<T>::calculateDiffuseReflex(vec3<T> lightPos, vec3<T> normal, vec3<T> collisionPos) {
	vec3<T> lightDir = lightPos - collisionPos;
	lightDir.normalize();
	normal.normalize();
	return MAX(lightDir.dot(normal), 0);
}

template <class T>
cv::Vec<T,4> Object<T>::computeColor(PointLight<T> light, vec3<T> normal, vec3<T> viewerPos, vec3<T> collisionPos) {
	vec3<T> lightDir = light.position - collisionPos;
	lightDir.normalize();
	normal.normalize();

	vec3<T> reflect = lightDir*(-1) - normal * (-2.) * lightDir.dot(normal);
	vec3<T> viewerDir = viewerPos - collisionPos;
	viewerDir.normalize();

	vec3<T> halfAngle = viewerDir + lightDir;
	halfAngle.normalize();

	T diffuse =  MAX(lightDir.dot(normal), 0);

	T blinn = MAX(viewerDir.dot(reflect), 0);
	blinn = pow(blinn, 8);

	cv::Vec<T, 4> ambientColor = 0.1 * color;
	cv::Vec<T, 4> diffuseColor = diffuse * cv::Vec<T, 4>(light.color.x * color[0], light.color.y * color[1], light.color.z * color[2], 1);
	cv::Vec<T, 4> specularColor = color[3] * blinn * cv::Vec<T, 4>(light.color.x, light.color.y, light.color.z, 1);

	return ambientColor + diffuseColor + specularColor;
}

template class Object<typeT>;