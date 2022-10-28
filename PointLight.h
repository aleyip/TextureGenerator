#pragma once
#include "Light.h"
#include "CudaPointers.h"

#include <opencv2/opencv.hpp>

template<class T>
class PointLight : public Light<T>
{
public:
    vec3<T> position;

    PointLight() = default;
    PointLight(vec3<T> position, vec3<T> color) :position(position), Light<T>(color) {};

    void addLightEffectsCUDA(CudaPointers<T>& cp, vec3<T>& viewerPos, int count);
};

