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
__global__ void CylinderCollisor_kernel(T* out, T* collision, T* normal, const T* rayLight, const T radius, const T height, const T* position, const T* rotation, const int sizeList) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < sizeList) {
        T origin_ray[3] = { rayLight[6 * index], rayLight[6 * index + 1], rayLight[6 * index + 2] };
        T direction_ray[3] = { rayLight[6 * index + 3], rayLight[6 * index + 4], rayLight[6 * index + 5] };

        T rotRad[] = { rotation[0] * degToRad,rotation[1] * degToRad,rotation[2] * degToRad };

        T cx = cos(rotRad[0]), sx = sin(rotRad[0]), cy = cos(rotRad[1]), sy = sin(rotRad[1]), cz = cos(rotRad[2]), sz = sin(rotRad[2]);

        T newOrigin_ray[] = {
            cz * cy * origin_ray[0] + sz * cy * origin_ray[1] - sy * origin_ray[2],
            (cz * sy * sx - sz * cx) * origin_ray[0] + (sz * sy * sx + cz * cx) * origin_ray[1] + (cy * sx) * origin_ray[2],
            (cz * sy * cx + sz * sx) * origin_ray[0] + (sz * sy * cx - cz * sx) * origin_ray[1] + (cy * cx) * origin_ray[2] };

        T newDirection_ray[] = {
            cz* cy* direction_ray[0] + sz * cy * direction_ray[1] - sy * direction_ray[2],
            (cz * sy * sx - sz * cx)* direction_ray[0] + (sz * sy * sx + cz * cx) * direction_ray[1] + (cy * sx) * direction_ray[2],
            (cz * sy * cx + sz * sx)* direction_ray[0] + (sz * sy * cx - cz * sx) * direction_ray[1] + (cy * cx) * direction_ray[2] };

        T relOrigin_ray[] = { newOrigin_ray[0] - position[0],newOrigin_ray[1] - position[1],newOrigin_ray[2] - position[2] };

        T a = newDirection_ray[0] * newDirection_ray[0] + newDirection_ray[2] * newDirection_ray[2];
        T b = 2. * (newDirection_ray[0] * relOrigin_ray[0] + newDirection_ray[2] * relOrigin_ray[2]);
        T c = relOrigin_ray[0] * relOrigin_ray[0] + relOrigin_ray[2] * relOrigin_ray[2] - radius * radius;

        T delta = -4. * a * c + b * b;
        T dist = 10e10, distAux = -1;
        T vec[3] = { 0,0,0 };
        if (delta >= 0)
        {
            distAux = (-b - sqrt(delta)) / (2. * a);
            T yCol = relOrigin_ray[1] + distAux * newDirection_ray[1];

            if (yCol <= height && yCol >= -height && distAux < dist) {
                dist = distAux;
                vec[0] = relOrigin_ray[0] + dist * newDirection_ray[0];
                vec[2] = relOrigin_ray[2] + dist * newDirection_ray[2];
                T mod = vec[0] * vec[0] + vec[2] * vec[2];
                mod = 1. / sqrt(mod);
                vec[0] *= mod;
                vec[2] *= mod;
            }
        }
        {
            T yTarget = height;
            distAux = (yTarget - relOrigin_ray[1]) / newDirection_ray[1];
            if (distAux >= 0 && distAux < dist) {
                T xCol = relOrigin_ray[0] + distAux * newDirection_ray[0], zCol = relOrigin_ray[2] + distAux * newDirection_ray[2];
                if (xCol * xCol + zCol * zCol <= radius * radius) {
                    distAux = dist;
                    vec[0] = 0;
                    vec[1] = 1;
                    vec[2] = 0;
                }

            }
            distAux = (-yTarget - relOrigin_ray[1]) / newDirection_ray[1];
            if (distAux >= 0 && distAux < dist) {
                T xCol = relOrigin_ray[0] + distAux * newDirection_ray[0], zCol = relOrigin_ray[2] + distAux * newDirection_ray[2];
                if (xCol * xCol + zCol * zCol <= radius * radius) {
                    distAux = dist;
                    vec[0] = 0;
                    vec[1] = -1;
                    vec[2] = 0;
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
cudaError_t CylinderCollisor_wrapper(std::vector<T>& out, const T diameter, const T height, const vec3<T> position, const vec3<T> rotation, CudaPointers<T>& cp) {
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
        CylinderCollisor_kernel<T> << < blocksPerGrid, threadsPerBlock >> > (cp.d_out, cp.d_collision, cp.d_normal, cp.d_rayList, diameter/2, height/2, cp.d_position, cp.d_rotation, out.size());
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
    cudaError_t cudaStatus = CylinderCollisor_wrapper<T>(out, diameter, height, position, rotation, cp);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cylinder Collisor Failed\n");
    }
}

template <class T>
T Cylinder<T>::CheckCollision(RayLight<T> ray, vec3<T>& collision, vec3<T>& normal) {
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

    T dist = 10e10, distAux = -1;
    if (delta >= 0)
    {
        distAux = (-b - sqrt(delta)) / (2. * a);
        T yCol = relOrigin.y + distAux * newDirection.y;

        if (yCol <= height / 2. && yCol >= -height / 2.) {
            dist = distAux;
            normal = relOrigin + newDirection * dist;
            normal.y = 0;
            normal.normalize();
        }
    }
    {
        T yTarget = height / 2.;
        distAux = (yTarget - relOrigin.y) / newDirection.y;
        if (distAux >= 0 && distAux < dist) {
            T xCol = relOrigin.x + distAux * newDirection.x, zCol = relOrigin.z + distAux * newDirection.z;
            if (xCol * xCol + zCol * zCol <= radius * radius) {
                normal = vec3<T>(0, 1, 0);
                dist = distAux;
            }
        }
        distAux = (-yTarget - relOrigin.y) / newDirection.y;
        if (distAux >= 0 && distAux < dist) {
            T xCol = relOrigin.x + distAux * newDirection.x, zCol = relOrigin.z + distAux * newDirection.z;
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

template class Cylinder<double>;
template class Cylinder<float>;