#include "Cylinder.h"

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
__global__ void CylinderCollisor_kernel(T* out, const T* rayLight, const T radius, const T height, const T* position, const T* rotation, const int sizeList) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < sizeList) {
        T origin[3] = { rayLight[6 * index], rayLight[6 * index + 1], rayLight[6 * index + 2] };
        T direction[3] = { rayLight[6 * index + 3], rayLight[6 * index + 4], rayLight[6 * index + 5] };

        T dirD[] = { rotation[0] * degToRad,rotation[1] * degToRad,rotation[2] * degToRad };

        T cx = cos(dirD[0]), sx = sin(dirD[0]), cy = cos(dirD[1]), sy = sin(dirD[1]), cz = cos(dirD[2]), sz = sin(dirD[2]);

        T newOrigin[] = {
            cz * cy * origin[0] + sz * cy * origin[1] - sy * origin[2],
            (cz * sy * sx - sz * cx) * origin[0] + (sz * sy * sx + cz * cx) * origin[1] + (cy * sx) * origin[2],
            (cz * sy * cx + sz * sx) * origin[0] + (sz * sy * cx - cz * sx) * origin[1] + (cy * cx) * origin[2] };

        T newDirection[] = {
            cz* cy* direction[0] + sz * cy * direction[1] - sy * direction[2],
            (cz * sy * sx - sz * cx)* direction[0] + (sz * sy * sx + cz * cx) * direction[1] + (cy * sx) * direction[2],
            (cz * sy * cx + sz * sx)* direction[0] + (sz * sy * cx - cz * sx) * direction[1] + (cy * cx) * direction[2] };

        T relOrigin[] = { newOrigin[0] - position[0],newOrigin[1] - position[1],newOrigin[2] - position[2] };

        T a = newDirection[0] * newDirection[0] + newDirection[2] * newDirection[2];
        T b = 2. * (newDirection[0] * relOrigin[0] + newDirection[2] * relOrigin[2]);
        T c = relOrigin[0] * relOrigin[0] + relOrigin[2] * relOrigin[2] - radius * radius;

        T delta = -4. * a * c + b * b;
        if (delta < 0) goto topBottomCollision;
        {
            T dist = (-b - sqrt(delta)) / (2. * a);
            T yCol = relOrigin[1] + dist * newDirection[1];

            if (yCol <= height / 2. && yCol >= -height / 2.) {
                out[index] = dist;
                return;
            }
        }
    topBottomCollision:
        {
            T yTarget = height / 2.;
            T dist = (yTarget - relOrigin[1]) / newDirection[1];
            if (dist >= 0) {
                T xCol = relOrigin[0] + dist * newDirection[0], zCol = relOrigin[2] + dist * newDirection[2];
                if (xCol * xCol + zCol * zCol <= radius * radius) {
                    out[index] = dist;
                    return;
                }

            }
        }
        {
            T yTarget = height / 2.;
            T dist = (-yTarget - relOrigin[1]) / newDirection[1];
            if (dist >= 0) {
                T xCol = relOrigin[0] + dist * newDirection[0], zCol = relOrigin[2] + dist * newDirection[2];
                if (xCol * xCol + zCol * zCol <= radius * radius) {
                    out[index] = dist;
                    return;
                }

            }
        }
        out[index] = -1;
    }
}

template <class T>
cudaError_t CylinderCollisor_wrapper(std::vector<T>& out, const T radius, const T height, const vec3<T> position, const vec3<T> rotation, CudaPointers<T>& cp) {
    cudaError_t cudaStatus;

    cudaStatus = cudaMemcpy(cp.d_position, &position, sizeof(vec3<T>), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed in d_position Host to Device!\n");
        goto ErrorCollisor;
    }

    cudaStatus = cudaMemcpy(cp.d_rotation, &rotation, sizeof(vec3<T>), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed in d_position Host to Device!\n");
        goto ErrorCollisor;
    }

    // Executing kernel 
    {
        dim3 threadsPerBlock(1024);
        dim3 blocksPerGrid(ceil(double(out.size()) / double(threadsPerBlock.x)));
        CylinderCollisor_kernel<T> << < blocksPerGrid, threadsPerBlock >> > (cp.d_out, cp.d_rayList, radius, height, cp.d_position, cp.d_rotation, out.size());
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
void Cylinder<T>::CheckCollisionCuda(std::vector<T>& out, CudaPointers<T>& cp) {
    cudaError_t cudaStatus = CylinderCollisor_wrapper<T>(out, diameter / 2., height, position, rotation, cp);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Sphere Collisor Failed\n");
    }
}

template <class T>
T Cylinder<T>::CheckCollision(RayLight<T> ray) {
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
    T b = 2. * (newDirection.x * relOrigin.x + newDirection.z * relOrigin.z);
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
        if (dist >= 0) {
            T xCol = relOrigin.x + dist * newDirection.x, zCol = relOrigin.z + dist * newDirection.z;
            if (xCol * xCol + zCol * zCol <= radius * radius)
                return dist;
        }
    }
    {
        T yTarget = height / 2.;
        T dist = (-yTarget - relOrigin.y) / newDirection.y;
        if (dist >= 0) {
            T xCol = relOrigin.x + dist * newDirection.x, zCol = relOrigin.z + dist * newDirection.z;
            if (xCol * xCol + zCol * zCol <= radius * radius)
                return dist;
        }
    }
    return -1;
}

template class Cylinder<double>;
template class Cylinder<float>;