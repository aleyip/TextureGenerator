#include "Sphere.h"
#include "defines.h"

#include <cuda.h>
#include <cuda_runtime.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ inline void atomicAdd(double* address, double value) {
    unsigned long long oldval, newval, readback;
    oldval = __double_as_longlong(*address);
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);
    while ((readback = atomicCAS((unsigned long long*) address, oldval, newval)) != oldval) {
        oldval = readback;
        newval = __double_as_longlong(__longlong_as_double(oldval) + value);
    }
}
#endif

//Vetor dir deve estar normalizado!!!!
template<class T>
__global__ void SphereCollisor_kernel(T* out, T* collision, T* normal, int8_t* objHit, const T* rayLight, 
    const T radius, const T* position, const int sizeList, const int objIndex) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < sizeList) {
        T origin[3] = { rayLight[6 * index], rayLight[6 * index + 1], rayLight[6 * index + 2] };
        T dir[3] = { rayLight[6 * index + 3], rayLight[6 * index + 4], rayLight[6 * index + 5] };

        T vec[3] = { origin[0] - position[0], origin[1] - position[1], origin[2] - position[2] };

        T dist;
        {
            T a = 1.;
            T b = 2. * (dir[0] * vec[0] + dir[1] * vec[1] + dir[2] * vec[2]);
            T c = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2] - radius * radius;

            T delta = -4. * a * c + b * b;
            if (delta < 0.) {
                return;
            }
            dist = (-b - sqrt(delta)) / (2. * a);
        }

        if (dist < out[index]) {
            T vec[3];
            vec[0] = origin[0] + dist * dir[0];
            vec[1] = origin[1] + dist * dir[1];
            vec[2] = origin[2] + dist * dir[2];

            collision[3 * index] = vec[0];
            collision[3 * index + 1] = vec[1];
            collision[3 * index + 2] = vec[2];

            vec[0] = vec[0] - position[0];
            vec[1] = vec[1] - position[1];
            vec[2] = vec[2] - position[2];

            T mod = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
            mod = 1. / sqrt(mod);

            vec[0] *= mod;
            vec[1] *= mod;
            vec[2] *= mod;

            normal[3 * index] = vec[0];
            normal[3 * index + 1] = vec[1];
            normal[3 * index + 2] = vec[2];

            out[index] = dist;
            objHit[index] = objIndex;
        }
    }
}

template <class T>
cudaError_t SphereCollisor_wrapper(int count, const T radius, const vec3<T> position, CudaPointers<T>& cp, int8_t objindex) {
    cudaError_t cudaStatus;

    cudaStatus = cudaMemcpy(cp.d_position, &position, sizeof(vec3<T>), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed in d_position Host to Device!\n");
        goto ErrorCollisor;
    }

    // Executing kernel 
    {
        dim3 threadsPerBlock(1024);
        dim3 blocksPerGrid(ceil(double(count) / double(threadsPerBlock.x)));
        SphereCollisor_kernel<T> << < blocksPerGrid, threadsPerBlock >> > (cp.d_dist, cp.d_collision, cp.d_normal, cp.d_hitobject, cp.d_rayList,
            radius, cp.d_position, count, objindex);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Sphere Collisor kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorCollisor;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Sphere Collisor Kernel!\n", cudaStatus);
        goto ErrorCollisor;
    }

ErrorCollisor:
    return cudaStatus;
}

template <class T>
void Sphere<T>::CheckCollisionCuda(CudaPointers<T>& cp, int count, int8_t objindex) {
    cudaError_t cudaStatus = SphereCollisor_wrapper<T>(count, diameter / 2., position, cp, objindex);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Sphere Collisor Failed\n");
    }
}

template <class T>
T Sphere<T>::CheckCollision(RayLight<T> ray, vec3<T>& collision, vec3<T>& normal) {
    T radius = diameter / 2.;

    ray.direction.normalize();

    vec3<T> relOrigin = ray.origin - this->position;

    T a = 1.;
    T b = 2. * ray.direction.dot(relOrigin);
    T c = relOrigin.dot(relOrigin) - radius * radius;

    T delta = -4. * a * c + b * b;
    if (delta < 0.) {
        return -1;
    }

    T dist = (-b - sqrt(delta)) / (2. * a);
    collision = ray.origin + ray.direction * dist;
    normal = collision - this->position;
    normal.normalize();

    return dist;
}

template class Sphere<typeT>;