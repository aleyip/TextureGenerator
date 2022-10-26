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
__global__ void CubeCollisor_kernel(T* out, T* collision, T* normal, const T* rayLight, const T width, const T height, const T depth, const T* position, const T* rotation, const int sizeList) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < sizeList) {
        T origin_ray[3] = { rayLight[6 * index], rayLight[6 * index + 1], rayLight[6 * index + 2] };
        T direction_ray[3] = { rayLight[6 * index + 3], rayLight[6 * index + 4], rayLight[6 * index + 5] };

        //T dirD[] = { rotation[0] * degToRad,rotation[1] * degToRad,rotation[2] * degToRad };
        T rotRad[] = { rotation[0] * degToRad,rotation[1] * degToRad,rotation[2] * degToRad };

        T cx = cos(rotRad[0]), sx = sin(rotRad[0]), cy = cos(rotRad[1]), sy = sin(rotRad[1]), cz = cos(rotRad[2]), sz = sin(rotRad[2]);

        T newOrigin_ray[] = {
            cz * cy * origin_ray[0] + sz * cy * origin_ray[1] - sy * origin_ray[2],
            (cz * sy * sx - sz * cx) * origin_ray[0] + (sz * sy * sx + cz * cx) * origin_ray[1] + (cy * sx) * origin_ray[2],
            (cz * sy * cx + sz * sx) * origin_ray[0] + (sz * sy * cx - cz * sx) * origin_ray[1] + (cy * cx) * origin_ray[2] };

        T newDirection_ray[] = {
            cz * cy * direction_ray[0] + sz * cy * direction_ray[1] - sy * direction_ray[2],
            (cz * sy * sx - sz * cx) * direction_ray[0] + (sz * sy * sx + cz * cx) * direction_ray[1] + (cy * sx) * direction_ray[2],
            (cz * sy * cx + sz * sx) * direction_ray[0] + (sz * sy * cx - cz * sx) * direction_ray[1] + (cy * cx) * direction_ray[2] };

        T relOrigin_ray[] = { newOrigin_ray[0] - position[0],newOrigin_ray[1] - position[1],newOrigin_ray[2] - position[2] };

        T dist = 10e10;
        T distAux = -1;
        T vec[3] = { 0,0,0 };
        {
            distAux = (depth - relOrigin_ray[0]) / newDirection_ray[0];
            if (distAux >= 0 && distAux < dist) {
                T yCol = relOrigin_ray[1] + distAux * newDirection_ray[1], zCol = relOrigin_ray[2] + distAux * newDirection_ray[2];
                if (yCol >= -height && yCol <= height &&
                    zCol >= -width && zCol <= width) {
                    dist = distAux;
                    vec[0] = 1;
                }
            }

            distAux = (-depth - relOrigin_ray[0]) / newDirection_ray[0];
            if (distAux >= 0 && distAux < dist) {
                T yCol = relOrigin_ray[1] + distAux * newDirection_ray[1], zCol = relOrigin_ray[2] + distAux * newDirection_ray[2];
                if (yCol >= -height && yCol <= height &&
                    zCol >= -width && zCol <= width) {
                    dist = distAux;
                    vec[0] = -1;
                }
            }
        }
        {
            distAux = (height - relOrigin_ray[1]) / newDirection_ray[1];
            if (distAux >= 0 && distAux < dist) {
                T xCol = relOrigin_ray[0] + distAux * newDirection_ray[0], zCol = relOrigin_ray[2] + distAux * newDirection_ray[2];
                if (xCol >= -depth && xCol <= depth &&
                    zCol >= -width && zCol <= width) {
                    dist = distAux;
                    vec[0] = 0;
                    vec[1] = 1;
                }
            }

            distAux = (-height - relOrigin_ray[1]) / newDirection_ray[1];
            if (distAux >= 0 && distAux < dist) {
                T xCol = relOrigin_ray[0] + distAux * newDirection_ray[0], zCol = relOrigin_ray[2] + distAux * newDirection_ray[2];
                if (xCol >= -depth && xCol <= depth &&
                    zCol >= -width && zCol <= width) {
                    dist = distAux;
                    vec[0] = 0;
                    vec[1] = -1;
                }
            }
        }
        {
            distAux = (width - relOrigin_ray[2]) / newDirection_ray[2];
            if (distAux >= 0 && distAux < dist) {
                T xCol = relOrigin_ray[0] + distAux * newDirection_ray[0], yCol = relOrigin_ray[1] + distAux * newDirection_ray[1];
                if (xCol >= -depth && xCol <= depth &&
                    yCol >= -height && yCol <= height) {
                    dist = distAux;
                    vec[0] = 0;
                    vec[1] = 0;
                    vec[2] = 1;
                }
            }

            distAux = (-width - relOrigin_ray[2]) / newDirection_ray[2];
            if (distAux >= 0 && distAux < dist) {
                T xCol = relOrigin_ray[0] + distAux * newDirection_ray[0], yCol = relOrigin_ray[1] + distAux * newDirection_ray[1];
                if (xCol >= -depth && xCol <= depth &&
                    yCol >= -height && yCol <= height) {
                    dist = distAux;
                    vec[0] = 0;
                    vec[1] = 0;
                    vec[2] = -1;
                }
            }
        }
        if (dist == 10e10) {
            out[index] = -1;
            return;
        }

        out[index] = dist;
        normal[3 * index] = cz * cy * vec[0] + (cz * sy * sx - sz * cx) * vec[1] + (cz * sy * cx + sz * sx) * vec[2];
        normal[3 * index + 1] = sz * cy * vec[0] + (sz * sy * sx + cz * cx) * vec[1] + (sz * sy * cx - cz * sx) * vec[2];
        normal[3 * index + 2] = -sy * vec[0] + (cy * sx) * vec[1] + (cy * cx) * vec[2];
        collision[3 * index] = origin_ray[0] + dist * direction_ray[0];
        collision[3 * index + 1] = origin_ray[1] + dist * direction_ray[1];
        collision[3 * index + 2] = origin_ray[2] + dist * direction_ray[2];
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
        dim3 threadsPerBlock(512);
        dim3 blocksPerGrid(ceil(double(out.size()) / double(threadsPerBlock.x)));
        CubeCollisor_kernel<T> << < blocksPerGrid, threadsPerBlock >> > (cp.d_out, cp.d_collision, cp.d_normal, cp.d_rayList, width / 2, height / 2, depth / 2, cp.d_position, cp.d_rotation, out.size());
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cube Collisor kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorCollisor;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Cube Collisor Kernel!\n", cudaStatus);
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
        fprintf(stderr, "Cube Collisor Failed\n");
    }
}

template <class T>
T Cube<T>::CheckCollision(RayLight<T> ray, vec3<T>& collision, vec3<T>& normal) {
    ray.direction.normalize();

    vec3<T> dirD = rotation * degToRad;

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

    T dist = 10e10;
    T distAux = -1;
    normal = vec3<T>(0, 0, 0);
    {
        T xTarget = depth / 2.;
        distAux = (xTarget - relOrigin.x) / newDirection.x;
        if (distAux >= 0 && distAux < dist) {
            T yCol = relOrigin.y + distAux * newDirection.y, zCol = relOrigin.z + distAux * newDirection.z;
            if (yCol >= -height / 2 && yCol <= height / 2 &&
                zCol >= -width / 2 && zCol <= width / 2) {
                dist = distAux;
                normal = vec3<T>(1,0,0);
            }
        }

        distAux = (-xTarget - relOrigin.x) / newDirection.x;
        if (distAux >= 0 && distAux < dist) {
            T yCol = relOrigin.y + distAux * newDirection.y, zCol = relOrigin.z + distAux * newDirection.z;
            if (yCol >= -height / 2 && yCol <= height / 2 &&
                zCol >= -width / 2 && zCol <= width / 2) {
                dist = distAux;
                normal = vec3<T>(-1, 0, 0);
            }
        }
    }
    {
        T yTarget = height / 2.;
        distAux = (yTarget - relOrigin.y) / newDirection.y;
        if (distAux >= 0 && distAux < dist) {
            T xCol = relOrigin.x + distAux * newDirection.x, zCol = relOrigin.z + distAux * newDirection.z;
            if (xCol >= -depth / 2 && xCol <= depth / 2 &&
                zCol >= -width / 2 && zCol <= width / 2) {
                dist = distAux;
                normal = vec3<T>(0, 1, 0);
            }
        }

        distAux = (-yTarget - relOrigin.y) / newDirection.y;
        if (distAux >= 0 && distAux < dist) {
            T xCol = relOrigin.x + distAux * newDirection.x, zCol = relOrigin.z + distAux * newDirection.z;
            if (xCol >= -depth / 2 && xCol <= depth / 2 &&
                zCol >= -width / 2 && zCol <= width / 2) {
                dist = distAux;
                normal = vec3<T>(0, -1, 0);
            }
        }
    }
    {
        T zTarget = width / 2.;
        distAux = (zTarget - relOrigin.z) / newDirection.z;
        if (distAux >= 0 && distAux < dist) {
            T xCol = relOrigin.x + distAux * newDirection.x, yCol = relOrigin.y + distAux * newDirection.y;
            if (xCol >= -depth / 2 && xCol <= depth / 2 &&
                yCol >= -height / 2 && yCol <= height / 2) {
                dist = distAux;
                normal = vec3<T>(0, 0, 1);
            }
        }

        distAux = (-zTarget - relOrigin.z) / newDirection.z;
        if (distAux >= 0 && distAux < dist) {
            T xCol = relOrigin.x + distAux * newDirection.x, yCol = relOrigin.y + distAux * newDirection.y;
            if (xCol >= -depth / 2 && xCol <= depth / 2 &&
                yCol >= -height / 2 && yCol <= height / 2) {
                dist = distAux;
                normal = vec3<T>(0, -1, -1);
            }
        }
    }       

    if (dist == 10e10) {
        return -1;
    }

    collision = ray.origin + ray.direction * dist;
    normal = vec3<T>(
        cz * cy * normal.x + (cz * sy * sx - sz * cx) * normal.y + (cz * sy * cx + sz * sx) * normal.z,
        sz * cy * normal.x + (sz * sy * sx + cz * cx) * normal.y + (sz * sy * cx - cz * sx) * normal.z,
        -sy * normal.x + (cy * sx) * normal.y + (cy * cx) * normal.z);

    return dist;
}

template class Cube<double>;
template class Cube<float>;