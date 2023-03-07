#include "Cube.h"
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
__global__ void CubeCollisor_kernel(T* out, T* collision, T* normal, int8_t* objHit, const T* rayLight,
    const T width, const T height, const T depth, const T* position, const T* rotation, const int sizeList, const int objIndex) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < sizeList) {
        T origin_ray[3] = { rayLight[6 * index], rayLight[6 * index + 1], rayLight[6 * index + 2] };
        T direction_ray[3] = { rayLight[6 * index + 3], rayLight[6 * index + 4], rayLight[6 * index + 5] };

        T aux[] = { origin_ray[0] - position[0],origin_ray[1] - position[1],origin_ray[2] - position[2] };

        //T dirD[] = { rotation[0] * degToRad,rotation[1] * degToRad,rotation[2] * degToRad };
        T rotRad[] = { rotation[0] * (-degToRad),rotation[1] * (-degToRad),rotation[2] * (-degToRad) };

        T cx = cos(rotRad[0]), sx = sin(rotRad[0]), cy = cos(rotRad[1]), sy = sin(rotRad[1]), cz = cos(rotRad[2]), sz = sin(rotRad[2]);

        T newOrigin_ray[] = {
            cz * cy * aux[0] + sz * cy * aux[1] - sy * aux[2],
            (cz * sy * sx - sz * cx) * aux[0] + (sz * sy * sx + cz * cx) * aux[1] + (cy * sx) * aux[2],
            (cz * sy * cx + sz * sx) * aux[0] + (sz * sy * cx - cz * sx) * aux[1] + (cy * cx) * aux[2] };

        T newDirection_ray[] = {
            cz * cy * direction_ray[0] + sz * cy * direction_ray[1] - sy * direction_ray[2],
            (cz * sy * sx - sz * cx) * direction_ray[0] + (sz * sy * sx + cz * cx) * direction_ray[1] + (cy * sx) * direction_ray[2],
            (cz * sy * cx + sz * sx) * direction_ray[0] + (sz * sy * cx - cz * sx) * direction_ray[1] + (cy * cx) * direction_ray[2] };

        T dist = 10e10;
        T distAux = -1;
        {
            distAux = (depth - newOrigin_ray[0]) / newDirection_ray[0];
            if (distAux >= 0 && distAux < dist) {
                T yCol = newOrigin_ray[1] + distAux * newDirection_ray[1], zCol = newOrigin_ray[2] + distAux * newDirection_ray[2];
                if (yCol >= -height && yCol <= height &&
                    zCol >= -width && zCol <= width) {
                    dist = distAux;
                    aux[0] = 1;
                    aux[1] = 0;
                    aux[2] = 0;
                }
            }

            distAux = (-depth - newOrigin_ray[0]) / newDirection_ray[0];
            if (distAux >= 0 && distAux < dist) {
                T yCol = newOrigin_ray[1] + distAux * newDirection_ray[1], zCol = newOrigin_ray[2] + distAux * newDirection_ray[2];
                if (yCol >= -height && yCol <= height &&
                    zCol >= -width && zCol <= width) {
                    dist = distAux;
                    aux[0] = -1;
                    aux[1] = 0;
                    aux[2] = 0;
                }
            }
        }
        {
            distAux = (height - newOrigin_ray[1]) / newDirection_ray[1];
            if (distAux >= 0 && distAux < dist) {
                T xCol = newOrigin_ray[0] + distAux * newDirection_ray[0], zCol = newOrigin_ray[2] + distAux * newDirection_ray[2];
                if (xCol >= -depth && xCol <= depth &&
                    zCol >= -width && zCol <= width) {
                    dist = distAux;
                    aux[0] = 0;
                    aux[1] = 1;
                    aux[2] = 0;
                }
            }

            distAux = (-height - newOrigin_ray[1]) / newDirection_ray[1];
            if (distAux >= 0 && distAux < dist) {
                T xCol = newOrigin_ray[0] + distAux * newDirection_ray[0], zCol = newOrigin_ray[2] + distAux * newDirection_ray[2];
                if (xCol >= -depth && xCol <= depth &&
                    zCol >= -width && zCol <= width) {
                    dist = distAux;
                    aux[0] = 0;
                    aux[1] = -1;
                    aux[2] = 0;
                }
            }
        }
        {
            distAux = (width - newOrigin_ray[2]) / newDirection_ray[2];
            if (distAux >= 0 && distAux < dist) {
                T xCol = newOrigin_ray[0] + distAux * newDirection_ray[0], yCol = newOrigin_ray[1] + distAux * newDirection_ray[1];
                if (xCol >= -depth && xCol <= depth &&
                    yCol >= -height && yCol <= height) {
                    dist = distAux;
                    aux[0] = 0;
                    aux[1] = 0;
                    aux[2] = 1;
                }
            }

            distAux = (-width - newOrigin_ray[2]) / newDirection_ray[2];
            if (distAux >= 0 && distAux < dist) {
                T xCol = newOrigin_ray[0] + distAux * newDirection_ray[0], yCol = newOrigin_ray[1] + distAux * newDirection_ray[1];
                if (xCol >= -depth && xCol <= depth &&
                    yCol >= -height && yCol <= height) {
                    dist = distAux;
                    aux[0] = 0;
                    aux[1] = 0;
                    aux[2] = -1;
                }
            }
        }

        if (dist < out[index]) 
        {
            out[index] = dist;
            objHit[index] = objIndex;
            normal[3 * index] = cz * cy * aux[0] + (cz * sy * sx - sz * cx) * aux[1] + (cz * sy * cx + sz * sx) * aux[2];
            normal[3 * index + 1] = sz * cy * aux[0] + (sz * sy * sx + cz * cx) * aux[1] + (sz * sy * cx - cz * sx) * aux[2];
            normal[3 * index + 2] = -sy * aux[0] + (cy * sx) * aux[1] + (cy * cx) * aux[2];
            collision[3 * index] = origin_ray[0] + dist * direction_ray[0];
            collision[3 * index + 1] = origin_ray[1] + dist * direction_ray[1];
            collision[3 * index + 2] = origin_ray[2] + dist * direction_ray[2];
        }
    }
}

template <class T>
cudaError_t CubeCollisor_wrapper(int count, const T width, const T height, const T depth, const vec3<T> position, const vec3<T> rotation, CudaPointers<T>& cp, int8_t objindex) {
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
        dim3 blocksPerGrid(ceil(double(count) / double(threadsPerBlock.x)));
        CubeCollisor_kernel<T> << < blocksPerGrid, threadsPerBlock >> > (cp.d_distList, cp.d_collisionList, cp.d_normalList, cp.d_hitobjectList, cp.d_rayList,
            width / 2, height / 2, depth / 2, cp.d_position, cp.d_rotation, count, objindex);
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
ErrorCollisor:
    return cudaStatus;
}

template <class T>
void Cube<T>::CheckCollisionCuda(CudaPointers<T>& cp, int count, int8_t objindex) {
    cudaError_t cudaStatus = CubeCollisor_wrapper<T>(count, width, height, depth, position, rotation, cp, objindex);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cube Collisor Failed\n");
    }
}

template <class T>
T Cube<T>::CheckCollision(RayLight<T> ray, vec3<T>& collision, vec3<T>& normal) {
    ray.direction.normalize();

    vec3<T> dirD = rotation * (-degToRad);

    T cx = cos(dirD.x), sx = sin(dirD.x), cy = cos(dirD.y), sy = sin(dirD.y), cz = cos(dirD.z), sz = sin(dirD.z);

    vec3<T>  relOrigin = ray.origin - this->position;

    vec3<T> newOrigin = vec3<T>(
        cz * cy * relOrigin.x + sz * cy * relOrigin.y - sy * relOrigin.z,
        (cz * sy * sx - sz * cx) * relOrigin.x + (sz * sy * sx + cz * cx) * relOrigin.y + (cy * sx) * relOrigin.z,
        (cz * sy * cx + sz * sx) * relOrigin.x + (sz * sy * cx - cz * sx) * relOrigin.y + (cy * cx) * relOrigin.z);

    vec3<T>  newDirection = vec3<T>(
        cz * cy * ray.direction.x + sz * cy * ray.direction.y - sy * ray.direction.z,
        (cz * sy * sx - sz * cx) * ray.direction.x + (sz * sy * sx + cz * cx) * ray.direction.y + (cy * sx) * ray.direction.z,
        (cz * sy * cx + sz * sx) * ray.direction.x + (sz * sy * cx - cz * sx) * ray.direction.y + (cy * cx) * ray.direction.z);

    T dist = 10e10;
    T distAux = -1;
    normal = vec3<T>(0, 0, 0);
    {
        T xTarget = depth / 2.;
        distAux = (xTarget - newOrigin.x) / newDirection.x;
        if (distAux >= 0 && distAux < dist) {
            T yCol = newOrigin.y + distAux * newDirection.y, zCol = newOrigin.z + distAux * newDirection.z;
            if (yCol >= -height / 2 && yCol <= height / 2 &&
                zCol >= -width / 2 && zCol <= width / 2) {
                dist = distAux;
                normal = vec3<T>(1, 0, 0);
            }
        }

        distAux = (-xTarget - newOrigin.x) / newDirection.x;
        if (distAux >= 0 && distAux < dist) {
            T yCol = newOrigin.y + distAux * newDirection.y, zCol = newOrigin.z + distAux * newDirection.z;
            if (yCol >= -height / 2 && yCol <= height / 2 &&
                zCol >= -width / 2 && zCol <= width / 2) {
                dist = distAux;
                normal = vec3<T>(-1, 0, 0);
            }
        }
    }
    {
        T yTarget = height / 2.;
        distAux = (yTarget - newOrigin.y) / newDirection.y;
        if (distAux >= 0 && distAux < dist) {
            T xCol = newOrigin.x + distAux * newDirection.x, zCol = newOrigin.z + distAux * newDirection.z;
            if (xCol >= -depth / 2 && xCol <= depth / 2 &&
                zCol >= -width / 2 && zCol <= width / 2) {
                dist = distAux;
                normal = vec3<T>(0, 1, 0);
            }
        }

        distAux = (-yTarget - newOrigin.y) / newDirection.y;
        if (distAux >= 0 && distAux < dist) {
            T xCol = newOrigin.x + distAux * newDirection.x, zCol = newOrigin.z + distAux * newDirection.z;
            if (xCol >= -depth / 2 && xCol <= depth / 2 &&
                zCol >= -width / 2 && zCol <= width / 2) {
                dist = distAux;
                normal = vec3<T>(0, -1, 0);
            }
        }
    }
    {
        T zTarget = width / 2.;
        distAux = (zTarget - newOrigin.z) / newDirection.z;
        if (distAux >= 0 && distAux < dist) {
            T xCol = newOrigin.x + distAux * newDirection.x, yCol = newOrigin.y + distAux * newDirection.y;
            if (xCol >= -depth / 2 && xCol <= depth / 2 &&
                yCol >= -height / 2 && yCol <= height / 2) {
                dist = distAux;
                normal = vec3<T>(0, 0, 1);
            }
        }

        distAux = (-zTarget - newOrigin.z) / newDirection.z;
        if (distAux >= 0 && distAux < dist) {
            T xCol = newOrigin.x + distAux * newDirection.x, yCol = newOrigin.y + distAux * newDirection.y;
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

template <class T>
void Cube<T>::Report() {
    printf("Cube: Position: %.2f %.2f %.2f Rotation: %.2f %.2f %.2f Color: %.2f %.2f %.2f Width: %.2f Height: %.2f Depth: %.2f\n", position.x, position.y, position.z, rotation.x,
        rotation.y, rotation.z, color[0], color[1], color[2], width, height, depth);
}

template class Cube<typeT>;