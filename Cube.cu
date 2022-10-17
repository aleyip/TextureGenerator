#include "Cube.h"

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
__global__ void CubeCollisor_kernel(T* out, const T* rayLight, const T width, const T height, const T depth, const T* position, const T* rotation, const int sizeList) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < sizeList) {
        T origin[3] = { rayLight[6 * index], rayLight[6 * index + 1], rayLight[6 * index + 2] };
        T direction[3] = { rayLight[6 * index + 3], rayLight[6 * index + 4], rayLight[6 * index + 5] };

        //T dirD[] = { rotation[0] * degToRad,rotation[1] * degToRad,rotation[2] * degToRad };
        T dirD[] = { rotation[0] * degToRad,rotation[1] * degToRad,rotation[2] * degToRad };

        T cx = cos(dirD[0]), sx = sin(dirD[0]), cy = cos(dirD[1]), sy = sin(dirD[1]), cz = cos(dirD[2]), sz = sin(dirD[2]);

        T newOrigin[] = {
            cz * cy * origin[0] + sz * cy * origin[1] - sy * origin[2],
            (cz * sy * sx - sz * cx) * origin[0] + (sz * sy * sx + cz * cx) * origin[1] + (cy * sx) * origin[2],
            (cz * sy * cx + sz * sx) * origin[0] + (sz * sy * cx - cz * sx) * origin[1] + (cy * cx) * origin[2] };

        T newDirection[] = {
            cz * cy * direction[0] + sz * cy * direction[1] - sy * direction[2],
            (cz * sy * sx - sz * cx) * direction[0] + (sz * sy * sx + cz * cx) * direction[1] + (cy * sx) * direction[2],
            (cz * sy * cx + sz * sx) * direction[0] + (sz * sy * cx - cz * sx) * direction[1] + (cy * cx) * direction[2] };

        T relOrigin[] = { newOrigin[0] - position[0],newOrigin[1] - position[1],newOrigin[2] - position[2] };

        T dist = -1;
        {
            T xTarget = depth;
            dist = (xTarget - relOrigin[0]) / newDirection[0];
            if (dist >= 0) {
                T yCol = relOrigin[1] + dist * newDirection[1], zCol = relOrigin[2] + dist * newDirection[2];
                if (yCol >= -height && yCol <= height &&
                    zCol >= -width && zCol <= width) {
                    out[index] = dist;
                    return;
                }
            }
        }
        {
            T xTarget = depth;
            dist = (-xTarget - relOrigin[0]) / newDirection[0];
            if (dist >= 0) {
                T yCol = relOrigin[1] + dist * newDirection[1], zCol = relOrigin[2] + dist * newDirection[2];
                if (yCol >= -height && yCol <= height &&
                    zCol >= -width && zCol <= width) {
                    out[index] = dist;
                    return;
                }
            }
        }
        {
            T yTarget = height;
            dist = (yTarget - relOrigin[1]) / newDirection[1];
            if (dist >= 0) {
                T xCol = relOrigin[0] + dist * newDirection[0], zCol = relOrigin[2] + dist * newDirection[2];
                if (xCol >= -depth && xCol <= depth &&
                    zCol >= -width && zCol <= width) {
                    out[index] = dist;
                    return;
                }
            }
        }
        {
            T yTarget = height;
            dist = (-yTarget - relOrigin[1]) / newDirection[1];
            if (dist >= 0) {
                T xCol = relOrigin[0] + dist * newDirection[0], zCol = relOrigin[2] + dist * newDirection[2];
                if (xCol >= -depth && xCol <= depth &&
                    zCol >= -width && zCol <= width) {
                    out[index] = dist;
                    return;
                }
            }
        }
        {
            T zTarget = width;
            dist = (zTarget - relOrigin[2]) / newDirection[2];
            if (dist >= 0) {
                T xCol = relOrigin[0] + dist * newDirection[0], yCol = relOrigin[1] + dist * newDirection[1];
                if (xCol >= -depth && xCol <= depth &&
                    yCol >= -height && yCol <= height) {
                    out[index] = dist;
                    return;
                }
            }
        }
        {
            T zTarget = width;
            dist = (-zTarget - relOrigin[2]) / newDirection[2];
            if (dist >= 0) {
                T xCol = relOrigin[0] + dist * newDirection[0], yCol = relOrigin[1] + dist * newDirection[1];
                if (xCol >= -depth && xCol <= depth &&
                    yCol >= -height && yCol <= height) {
                    out[index] = dist;
                    return;
                }
            }
        }
        out[index] = -1;
    }
}

template <class T>
cudaError_t CubeCollisor_wrapper(std::vector<T>& out, const T width, const T height, const T depth, const vec3<T> position, const vec3<T> rotation, CudaPointers<T>& cp) {
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
        CubeCollisor_kernel<T> << < blocksPerGrid, threadsPerBlock >> > (cp.d_out, cp.d_rayList, width / 2, height / 2, depth / 2, cp.d_position, cp.d_rotation, out.size());
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
void Cube<T>::CheckCollisionCuda(std::vector<T>& out, CudaPointers<T>& cp) {
    cudaError_t cudaStatus = CubeCollisor_wrapper<T>(out, width, height, depth, position, rotation, cp);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Sphere Collisor Failed\n");
    }
}

template <class T>
T Cube<T>::CheckCollision(RayLight<T> ray) {
    T mod = ray.direction.dot(ray.direction);
    if (mod != 1.) {
        mod = sqrt(mod);
        ray.direction = ray.direction * (1. / mod);
    }

    vec3<T> dirD = this->rotation * degToRad;

    T cx = cos(dirD.x), sx = sin(dirD.x), cy = cos(dirD.y), sy = sin(dirD.y), cz = cos(dirD.z), sz = sin(dirD.z);

    vec3<T> newOrigin = vec3<T>(
        cz * cy * ray.origin.x + sz * cy * ray.origin.y - sy * ray.origin.z,
        (cz * sy * sx - sz * cx) * ray.origin.x + (sz * sy * sx + cz * cx) * ray.origin.y + (cy * sx) * ray.origin.z,
        (cz * sy * cx + sz * sx) * ray.origin.x + (sz * sy * cx - cz * sx) * ray.origin.y + (cy * cx) * ray.origin.z);

    vec3<T>  newDirection = vec3<T>(
        cz * cy * ray.direction.x + sz * cy * ray.direction.y - sy * ray.direction.z,
        (cz * sy * sx - sz * cx) * ray.direction.x + (sz * sy * sx + cz * cx) * ray.direction.y + (cy * sx) * ray.direction.z,
        (cz * sy * cx + sz * sx) * ray.direction.x + (sz * sy * cx - cz * sx) * ray.direction.y + (cy * cx) * ray.direction.z);

    vec3<T>  relOrigin = newOrigin - this->position;

    T dist = -1;
    {
        T xTarget = depth / 2.;
        dist = (xTarget - relOrigin.x) / newDirection.x;
        if (dist >= 0) {
            T yCol = relOrigin.y + dist * newDirection.y, zCol = relOrigin.z + dist * newDirection.z;
            if (yCol >= -height / 2 && yCol <= height / 2 &&
                zCol >= -width / 2 && zCol <= width / 2)
                return dist;
        }
    }
    {
        T xTarget = depth / 2.;
        dist = (-xTarget - relOrigin.x) / newDirection.x;
        if (dist >= 0) {
            T yCol = relOrigin.y + dist * newDirection.y, zCol = relOrigin.z + dist * newDirection.z;
            if (yCol >= -height / 2 && yCol <= height / 2 &&
                zCol >= -width / 2 && zCol <= width / 2)
                return dist;
        }
    }
    {
        T yTarget = height / 2.;
        dist = (yTarget - relOrigin.y) / newDirection.y;
        if (dist >= 0) {
            T xCol = relOrigin.x + dist * newDirection.x, zCol = relOrigin.z + dist * newDirection.z;
            if (xCol >= -depth / 2 && xCol <= depth / 2 &&
                zCol >= -width / 2 && zCol <= width / 2)
                return dist;
        }
    }
    {
        T yTarget = height / 2.;
        dist = (-yTarget - relOrigin.y) / newDirection.y;
        if (dist >= 0) {
            T xCol = relOrigin.x + dist * newDirection.x, zCol = relOrigin.z + dist * newDirection.z;
            if (xCol >= -depth / 2 && xCol <= depth / 2 &&
                zCol >= -width / 2 && zCol <= width / 2)
                return dist;
        }
    }
    {
        T zTarget = width / 2.;
        dist = (zTarget - relOrigin.z) / newDirection.z;
        if (dist >= 0) {
            T xCol = relOrigin.x + dist * newDirection.x, yCol = relOrigin.y + dist * newDirection.y;
            if (xCol >= -depth / 2 && xCol <= depth / 2 &&
                yCol >= -height / 2 && yCol <= height / 2)
                return dist;
        }
    }
    {
        T zTarget = width / 2.;
        dist = (-zTarget - relOrigin.z) / newDirection.z;
        if (dist >= 0) {
            T xCol = relOrigin.x + dist * newDirection.x, yCol = relOrigin.y + dist * newDirection.y;
            if (xCol >= -depth / 2 && xCol <= depth / 2 &&
                yCol >= -height / 2 && yCol <= height / 2)
                return dist;
        }
    }
    return  -1;
}

template class Cube<double>;
template class Cube<float>;