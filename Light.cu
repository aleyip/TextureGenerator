#include "Light.h"
#include "defines.h"

#include <cuda.h>
#include <cuda_runtime.h>

template<class T>
__global__ void SetAmbientLight_kernel(T* out, int8_t* objHit, T* objColor, const T ambient, const int sizeList) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < sizeList) {
        int obj = objHit[index];
        T* pixel = out + 4 * index;
        if (obj == -1) {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
            pixel[3] = 0;
        }
        else {
            T* objColor_ptr = objColor + 4 * obj;

            pixel[0] = ambient * objColor_ptr[0];
            pixel[1] = ambient * objColor_ptr[1];
            pixel[2] = ambient * objColor_ptr[2];
            pixel[3] = 1;
        }
    }
}

template <class T>
cudaError_t SetAmbienteLight_wrapper(int count, T ambientValue, CudaPointers<T>& cp) {
    cudaError_t cudaStatus;

    // Executing kernel 
    {
        dim3 threadsPerBlock(1024);
        dim3 blocksPerGrid(ceil(double(count) / double(threadsPerBlock.x)));
        SetAmbientLight_kernel<T> << < blocksPerGrid, threadsPerBlock >> > (cp.d_colorList, cp.d_hitobjectList, cp.d_objcolorList, ambientValue, count);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Ambient Light kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorAmbient;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Ambient Light Kernel!\n", cudaStatus);
        goto ErrorAmbient;
    }
ErrorAmbient:
    return cudaStatus;
}

template<class T>
static void Light<T>::setAmbientLightCUDA(CudaPointers<T>& cp, T ambientValue, int count) {
    cudaError_t cudaStatus = SetAmbienteLight_wrapper<T>(count, ambientValue, cp);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Ambient Light Failed\n");
    }
}

template<class T>
static cv::Vec<T, 4> Light<T>::setAmbientLight(Object<T>& obj, T ambientValue) {
    cv::Vec<T, 4> color = ambientValue * obj.color;
    color[3] = 1;
    return color;
}

template class Light<typeT>;