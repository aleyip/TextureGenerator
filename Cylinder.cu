#include "Cylinder.h"
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
__global__ void CylinderCollisor_kernel(T* out, T* collision, T* normal, int8_t* objHit, const T* rayLight,
    const T radius, const T height, const T* position, const T* rotation, const int sizeList, const int objIndex) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < sizeList) {
        T origin_ray[3] = { rayLight[6 * index], rayLight[6 * index + 1], rayLight[6 * index + 2] };
        T direction_ray[3] = { rayLight[6 * index + 3], rayLight[6 * index + 4], rayLight[6 * index + 5] };

        T rotRad[] = { rotation[0] * (-degToRad),rotation[1] * (-degToRad),rotation[2] * (-degToRad) };

        T aux[] = { origin_ray[0] - position[0],origin_ray[1] - position[1],origin_ray[2] - position[2] };

        T cx = cos(rotRad[0]), sx = sin(rotRad[0]), cy = cos(rotRad[1]), sy = sin(rotRad[1]), cz = cos(rotRad[2]), sz = sin(rotRad[2]);

        T newOrigin_ray[] = {
            cz * cy * aux[0] + sz * cy * aux[1] - sy * aux[2],
            (cz * sy * sx - sz * cx) * aux[0] + (sz * sy * sx + cz * cx) * aux[1] + (cy * sx) * aux[2],
            (cz * sy * cx + sz * sx) * aux[0] + (sz * sy * cx - cz * sx) * aux[1] + (cy * cx) * aux[2] };

        T newDirection_ray[] = {
            cz* cy* direction_ray[0] + sz * cy * direction_ray[1] - sy * direction_ray[2],
            (cz * sy * sx - sz * cx)* direction_ray[0] + (sz * sy * sx + cz * cx) * direction_ray[1] + (cy * sx) * direction_ray[2],
            (cz * sy * cx + sz * sx)* direction_ray[0] + (sz * sy * cx - cz * sx) * direction_ray[1] + (cy * cx) * direction_ray[2] };

        T a = newDirection_ray[0] * newDirection_ray[0] + newDirection_ray[2] * newDirection_ray[2];
        T b = 2. * (newDirection_ray[0] * newOrigin_ray[0] + newDirection_ray[2] * newOrigin_ray[2]);
        T c = newOrigin_ray[0] * newOrigin_ray[0] + newOrigin_ray[2] * newOrigin_ray[2] - radius * radius;

        T delta = -4. * a * c + b * b;
        T dist = 10e10, distAux = -1;
        if (delta >= 0)
        {
            distAux = (-b - sqrt(delta)) / (2. * a);
            T yCol = newOrigin_ray[1] + distAux * newDirection_ray[1];

            if (yCol <= height && yCol >= -height && distAux < dist) {
                dist = distAux;
                aux[0] = newOrigin_ray[0] + dist * newDirection_ray[0];
                aux[2] = newOrigin_ray[2] + dist * newDirection_ray[2];
                T mod = aux[0] * aux[0] + aux[2] * aux[2];
                mod = 1. / sqrt(mod);
                aux[0] *= mod;
                aux[1] = 0;
                aux[2] *= mod;
            }
        }
        {
            distAux = (height - newOrigin_ray[1]) / newDirection_ray[1];
            if (distAux >= 0 && distAux < dist) {
                T xCol = newOrigin_ray[0] + distAux * newDirection_ray[0], zCol = newOrigin_ray[2] + distAux * newDirection_ray[2];
                if (xCol * xCol + zCol * zCol <= radius * radius) {
                    dist = distAux;
                    aux[0] = 0;
                    aux[1] = 1;
                    aux[2] = 0;
                }

            }
            distAux = (-height - newOrigin_ray[1]) / newDirection_ray[1];
            if (distAux >= 0 && distAux < dist) {
                T xCol = newOrigin_ray[0] + distAux * newDirection_ray[0], zCol = newOrigin_ray[2] + distAux * newDirection_ray[2];
                if (xCol * xCol + zCol * zCol <= radius * radius) {
                    dist = distAux;
                    aux[0] = 0;
                    aux[1] = -1;
                    aux[2] = 0;
                }

            }
        }
        if (dist < out[index]) {
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
cudaError_t CylinderCollisor_wrapper(int count, const T diameter, const T height, const vec3<T> position, const vec3<T> rotation, CudaPointers<T>& cp, int8_t objindex) {
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
        CylinderCollisor_kernel<T> << < blocksPerGrid, threadsPerBlock >> > (cp.d_dist, cp.d_collision, cp.d_normal, cp.d_hitobject, cp.d_rayList,
            diameter/2, height/2, cp.d_position, cp.d_rotation, count, objindex);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cylinder Collisor kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorCollisor;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Cylinder Collisor Kernel!\n", cudaStatus);
        goto ErrorCollisor;
    }

ErrorCollisor:
    return cudaStatus;
}

template <class T>
void Cylinder<T>::CheckCollisionCuda(CudaPointers<T>& cp, int count, int8_t objindex) {
    cudaError_t cudaStatus = CylinderCollisor_wrapper<T>(count, diameter, height, position, rotation, cp, objindex);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cylinder Collisor Failed\n");
    }
}

template <class T>
T Cylinder<T>::CheckCollision(RayLight<T> ray, vec3<T>& collision, vec3<T>& normal) {
    T radius = diameter / 2.;

    ray.direction.normalize();

    vec3<T> dirD = this->rotation * (-degToRad);

    vec3<T> relOrigin = ray.origin - this->position;

    T cx = cos(dirD.x), sx = sin(dirD.x), cy = cos(dirD.y), sy = sin(dirD.y), cz = cos(dirD.z), sz = sin(dirD.z);

    vec3<T> newOrigin = vec3<T>(
        cz * cy * relOrigin.x + sz * cy * relOrigin.y - sy * relOrigin.z,
        (cz * sy * sx - sz * cx) * relOrigin.x + (sz * sy * sx + cz * cx) * relOrigin.y + (cy * sx) * relOrigin.z,
        (cz * sy * cx + sz * sx) * relOrigin.x + (sz * sy * cx - cz * sx) * relOrigin.y + (cy * cx) * relOrigin.z);

    vec3<T> newDirection = vec3<T>(
        (cz * cy) * ray.direction.x + sz * cy * ray.direction.y - sy * ray.direction.z,
        (cz * sy * sx - sz * cx) * ray.direction.x + (sz * sy * sx + cz * cx) * ray.direction.y + (cy * sx) * ray.direction.z,
        (cz * sy * cx + sz * sx) * ray.direction.x + (sz * sy * cx - cz * sx) * ray.direction.y + cy * cx * ray.direction.z);

    

    T a = newDirection.x * newDirection.x + newDirection.z * newDirection.z;
    T b = 2. * (newDirection.x * newOrigin.x + newDirection.z * newOrigin.z);
    T c = newOrigin.x * newOrigin.x + newOrigin.z * newOrigin.z - radius * radius;

    T delta = -4. * a * c + b * b;

    T dist = 10e10, distAux = -1;
    if (delta >= 0)
    {
        distAux = (-b - sqrt(delta)) / (2. * a);
        T yCol = newOrigin.y + distAux * newDirection.y;

        if (yCol <= height / 2. && yCol >= -height / 2.) {
            dist = distAux;
            normal = newOrigin + newDirection * dist;
            normal.y = 0;
            normal.normalize();
        }
    }
    {
        T yTarget = height / 2.;
        distAux = (yTarget - newOrigin.y) / newDirection.y;
        if (distAux >= 0 && distAux < dist) {
            T xCol = newOrigin.x + distAux * newDirection.x, zCol = newOrigin.z + distAux * newDirection.z;
            if (xCol * xCol + zCol * zCol <= radius * radius) {
                normal = vec3<T>(0, 1, 0);
                dist = distAux;
            }
        }
        distAux = (-yTarget - newOrigin.y) / newDirection.y;
        if (distAux >= 0 && distAux < dist) {
            T xCol = newOrigin.x + distAux * newDirection.x, zCol = newOrigin.z + distAux * newDirection.z;
            if (xCol * xCol + zCol * zCol <= radius * radius) {
                normal = vec3<T>(0, -1, 0);
                dist = distAux;
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

template class Cylinder<typeT>;