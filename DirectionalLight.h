#pragma once
#include "Light.h"
#include "CudaPointers.h"

#include <opencv2/opencv.hpp>

template<class T>
class DirectionalLight : public Light<T>
{
public:
    vec3<T> direction;

    DirectionalLight() = default;
    DirectionalLight(vec3<T> dir, vec3<T> color) :direction(dir), Light<T>(color) {};

    cv::Vec<T, 4> lightEffect(Object<T>& obj, vec3<T>& collision, vec3<T>& normal, vec3<T>& viewerPos);
    void addLightEffectsCUDA(CudaPointers<T>& cp, int count);
};
