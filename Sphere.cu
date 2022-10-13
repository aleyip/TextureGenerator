#include "Sphere.h"

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
__global__ void SphereCollisor_kernel(T* out, const T* rayLight, const T radius, const T* position, const int sizeList) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < sizeList) {
        double origin[3] = { rayLight[6 * index] - position[0], rayLight[6 * index + 1] - position[1], rayLight[6 * index + 2] - position[2] };
        double dir[3] = { rayLight[6 * index + 3], rayLight[6 * index + 4], rayLight[6 * index + 5] };

        T a = 1.;
        T b = 2. * (dir[0] * origin[0] + dir[1] * origin[1] + dir[2] * origin[2]);
        T c = origin[0] * origin[0] + origin[1] * origin[1] + origin[2] * origin[2] - radius * radius;

        T delta = -4. * a * c + b * b;
        if (delta < 0.) {
            out[index] = -1;
            return;
        }

        out[index] = (-b - sqrt(delta)) / (2. * a);
    }
}

template <class T>
cudaError_t SphereCollisor_wrapper(std::vector<T>& out, const T radius, const vec3<T> position, CudaPointers<T>& cp) {
    cudaError_t cudaStatus;

    //// Transfer data from host to device memory
    //cudaStatus = cudaMemcpy(cp.d_rayList, rayList.data(), sizeof(RayLight<T>) * rayList.size(), cudaMemcpyHostToDevice);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed in d_system Host to Device!\n");
    //    goto ErrorCollisor;
    //}

    ////Transfer data back to host memory
    //cudaStatus = cudaMemcpy(rayList.data(), cp.d_rayList, sizeof(RayLight<T>) * rayList.size(), cudaMemcpyDeviceToHost);
    //if (cudaStatus != cudaSuccess) {
    //	fprintf(stderr, "cudaMemcpy failed in d_out Device to Host!\n");
    //	goto ErrorCollisor;
    //}

    //for (int i = 0; i < 100; i++)
    //{
    //	printf("b%03d \t %f %f %f \t %f %f %f\n", i, rayList[i].origin.x, rayList[i].origin.y, rayList[i].origin.z, rayList[i].direction.x, rayList[i].direction.y, rayList[i].direction.z);
    //}

    cudaStatus = cudaMemcpy(cp.d_position, &position, sizeof(vec3<T>), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed in d_position Host to Device!\n");
        goto ErrorCollisor;
    }

    // Executing kernel 
    {
        dim3 threadsPerBlock(1024);
        dim3 blocksPerGrid(ceil(double(out.size()) / double(threadsPerBlock.x)));
        SphereCollisor_kernel<T> << < blocksPerGrid, threadsPerBlock >> > (cp.d_out, cp.d_rayList, radius, cp.d_position, out.size());
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

    // Transfer data back to host memory
    cudaStatus = cudaMemcpy(out.data(), cp.d_out, sizeof(T) * out.size(), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed in d_out Device to Host!\n");
        goto ErrorCollisor;
    }

ErrorCollisor:
    return cudaStatus;
}

template <class T>
void Sphere<T>::CheckCollisionCuda(std::vector<T>& out, CudaPointers<T>& cp) {
    cudaError_t cudaStatus = SphereCollisor_wrapper<T>(out, diameter / 2., position, cp);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Sphere Collisor Failed\n");
    }
}

template <class T>
T Sphere<T>::CheckCollision(RayLight<T> ray) {
    T radius = diameter / 2.;

    T mod = ray.direction.dot(ray.direction);
    if (mod != 1.) {
        mod = sqrt(mod);
        ray.direction = ray.direction * (1. / mod);
    }

    vec3<T> relOrigin = ray.origin - this->position;

    T a = 1.;
    T b = 2. * ray.direction.dot(relOrigin);
    T c = relOrigin.dot(relOrigin) - radius * radius;

    T delta = -4. * a * c + b * b;
    if (delta < 0.) return -1;

    return  (-b - sqrt(delta)) / (2. * a);
}

template class Sphere<double>;
template class Sphere<float>;