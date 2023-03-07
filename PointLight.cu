#include "PointLight.h"
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

template<class T>
__global__ void addLightEffects_kernel(T* out, T* collision, T* normal, int8_t* objHit, T* objColor, uint8_t* shininess,
    T* lightPos, T* lightCol, T* ray, const int sizeList) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < sizeList) {
        int obj = objHit[index];
        T* pixel = out + 4 * index;
        T* rayLight = ray + 6 * index;
        if (obj != -1) {
            T* objColor_ptr = objColor + 4 * obj;
            int shiny = shininess[obj];
            T* collision_ptr = collision + 3 * index;
            T* normal_ptr = normal + 3 * index;

            //Light Dir
            T aux_vec[] = { lightPos[0] - collision_ptr[0],lightPos[1] - collision_ptr[1],lightPos[2] - collision_ptr[2] };
            T aux_db = aux_vec[0] * aux_vec[0] + aux_vec[1] * aux_vec[1] + aux_vec[2] * aux_vec[2];
            aux_db = 1 / sqrt(aux_db);
            aux_vec[0] *= aux_db;
            aux_vec[1] *= aux_db;
            aux_vec[2] *= aux_db;

            aux_db = normal_ptr[0] * normal_ptr[0] + normal_ptr[1] * normal_ptr[1] + normal_ptr[2] * normal_ptr[2];
            aux_db = 1 / sqrt(aux_db);
            normal_ptr[0] *= aux_db;
            normal_ptr[1] *= aux_db;
            normal_ptr[2] *= aux_db;

            //lightDir.dot(normal)
            T diffuse = aux_vec[0] * normal_ptr[0] + aux_vec[1] * normal_ptr[1] + aux_vec[2] * normal_ptr[2];
            if (diffuse < 0) diffuse = 0;

            ////Reflect Vector
            //aux_db = aux_vec[0] * normal_ptr[0] + aux_vec[1] * normal_ptr[1] + aux_vec[2] * normal_ptr[2];
            //aux_vec[0] = aux_vec[0] - 2. * aux_db * normal[0];
            //aux_vec[1] = aux_vec[1] - 2. * aux_db * normal[1];
            //aux_vec[2] = aux_vec[2] - 2. * aux_db * normal[2];
            //aux_db = aux_vec[0] * aux_vec[0] + aux_vec[1] * aux_vec[1] + aux_vec[2] * aux_vec[2];
            //aux_db = 1 / sqrt(aux_db);
            //aux_vec[0] *= aux_db;
            //aux_vec[1] *= aux_db;
            //aux_vec[2] *= aux_db;

            //Viewer Dir
            //T aux_vec2[] = { rayLight[0] - collision_ptr[0], rayLight[1] - collision_ptr[1], rayLight[2] - collision_ptr[2] };
            T aux_vec2[] = { -rayLight[3], -rayLight[4], -rayLight[5] };
            aux_db = aux_vec2[0] * aux_vec2[0] + aux_vec2[1] * aux_vec2[1] + aux_vec2[2] * aux_vec2[2];
            aux_db = 1. / sqrt(aux_db);
            aux_vec2[0] *= aux_db;
            aux_vec2[1] *= aux_db;
            aux_vec2[2] *= aux_db;

            aux_vec[0] += aux_vec2[0];
            aux_vec[1] += aux_vec2[1];
            aux_vec[2] += aux_vec2[2];
            aux_db = aux_vec[0] * aux_vec[0] + aux_vec[1] * aux_vec[1] + aux_vec[2] * aux_vec[2];
            aux_db = 1. / sqrt(aux_db);
            aux_vec[0] *= aux_db;
            aux_vec[1] *= aux_db;
            aux_vec[2] *= aux_db;

            //vec3<T> halfAngle = viewerDir + lightDir;
            //halfAngle.normalize();

            //Ou halfDir e Normal
            ////T blinn = aux_vec[0] * aux_vec2[0] + aux_vec[1] * aux_vec2[1] + aux_vec[2] * aux_vec2[2];
            //T blinn = aux_vec[0] * aux_vec2[0] + aux_vec[1] * aux_vec2[1] + aux_vec[2] * aux_vec2[2];
            ////if (blinn < 0) blinn = 0;
            //blinn = pow(blinn,256);

            T blinn = aux_vec[0] * normal_ptr[0] + aux_vec[1] * normal_ptr[1] + aux_vec[2] * normal_ptr[2];
            if (blinn < 0) blinn = 0;
            blinn = pow(blinn, shiny);

            pixel[0] += (lightCol[0] * diffuse) * objColor_ptr[0] + lightCol[0] * blinn * objColor_ptr[3];
            pixel[1] += (lightCol[1] * diffuse) * objColor_ptr[1] + lightCol[1] * blinn * objColor_ptr[3];
            pixel[2] += (lightCol[2] * diffuse) * objColor_ptr[2] + lightCol[2] * blinn * objColor_ptr[3];
        }
    }
}

template <class T>
cudaError_t addLightEffects_wrapper(int count, vec3<T> lightColor, vec3<T> lightPos, CudaPointers<T>& cp) {
    cudaError_t cudaStatus;

    cudaStatus = cudaMemcpy(cp.d_position, &lightPos, sizeof(vec3<T>), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed in d_rotation Host to Device!\n");
        goto ErrorCollisor;
    }

    cudaStatus = cudaMemcpy(cp.d_color, &lightColor, sizeof(vec3<T>), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed in d_color Host to Device!\n");
        goto ErrorCollisor;
    }

    // Executing kernel 
    {
        dim3 threadsPerBlock(1024);
        dim3 blocksPerGrid(ceil(double(count) / double(threadsPerBlock.x)));
        addLightEffects_kernel<T> << < blocksPerGrid, threadsPerBlock >> > (cp.d_colorList, cp.d_collisionList, cp.d_normalList, cp.d_hitobjectList, cp.d_objcolorList, cp.d_objShinList,
            cp.d_position, cp.d_color, cp.d_rayList, count);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Add Light Effects kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorCollisor;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Add Light Effects Kernel!\n", cudaStatus);
        goto ErrorCollisor;
    }
ErrorCollisor:
    return cudaStatus;
}

template <class T>
void PointLight<T>::addLightEffectsCUDA(CudaPointers<T>& cp, int count) {
    cudaError_t cudaStatus = addLightEffects_wrapper<T>(count, color, this->position, cp);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Add Light Effects Failed\n");
    }
}

template <class T>
cv::Vec<T, 4> PointLight<T>::lightEffect(Object<T>& obj, vec3<T>& collision, vec3<T>& normal, vec3<T>& viewerPos) {
    vec3<T> lightDir = position - collision;
    lightDir.normalize();
    normal.normalize();

    vec3<T> reflect = lightDir * (-1) - normal * (-2.) * lightDir.dot(normal);
    vec3<T> viewerDir = viewerPos - collision;
    viewerDir.normalize();

    vec3<T> halfAngle = viewerDir + lightDir;
    halfAngle.normalize();

    T diffuse = MAX(lightDir.dot(normal), 0);

    T blinn = MAX(normal.dot(halfAngle), 0);
    blinn = pow(blinn, obj.specularShininness);

    cv::Vec<T, 4> diffuseColor = diffuse * cv::Vec<T, 4>(color.x * obj.color[0], color.y * obj.color[1], color.z * obj.color[2], 1);
    cv::Vec<T, 4> specularColor = obj.color[3] * blinn * cv::Vec<T, 4>(color.x, color.y, color.z, 1);

    return diffuseColor + specularColor;
}

template <class T>
void PointLight<T>::Report() {
    printf("Point Light: Color: %.2f %.2f %.2f Position: %.2f %.2f %.2f\n", color.x, color.y, color.z, position.x, position.y, position.z);
}

template class PointLight<typeT>;