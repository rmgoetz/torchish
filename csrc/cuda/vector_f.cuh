
#ifndef CUSTOM_CUDA_VECTOR_3D_FLOAT
#define CUSTOM_CUDA_VECTOR_3D_FLOAT

#include <array>
#include <cuda.h>
#include <cuda_runtime.h>

class cudVec3D_f
{
public:
    float data[3] = {0, 0, 0};

public:
    __device__ cudVec3D_f(float *value_ptr)
    {
        for (int i = 0; i < 3; i++)
        {
            data[i] = value_ptr[i];
        }
    }

    __device__ cudVec3D_f() {}

    // Addition
    __device__ inline cudVec3D_f add(const cudVec3D_f &other)
    {
        float result[3];
        for (int i = 0; i < 3; i++)
        {
            result[i] = data[i] + other.data[i];
        }
        return cudVec3D_f(result);
    }

    // Subtraction
    __device__ inline cudVec3D_f sub(const cudVec3D_f &other)
    {
        float result[3];
        for (int i = 0; i < 3; i++)
        {
            result[i] = data[i] - other.data[i];
        }
        return cudVec3D_f(result);
    }

    __device__ inline cudVec3D_f prod(const cudVec3D_f &other)
    {
        float result[3];
        result[0] = data[0] * other.data[0];
        result[1] = data[1] * other.data[1];
        result[2] = data[2] * other.data[2];
        return cudVec3D_f(result);
    }

    __device__ inline cudVec3D_f prod(const float &scalar)
    {
        float result[3];
        result[0] = data[0] * scalar;
        result[1] = data[1] * scalar;
        result[2] = data[2] * scalar;
        return cudVec3D_f(result);
    }

    // Cross product
    __device__ inline cudVec3D_f cross(const cudVec3D_f &other)
    {
        float result[3];
        result[0] = data[1] * other.data[2] - data[2] * other.data[1];
        result[1] = data[2] * other.data[0] - data[0] * other.data[2];
        result[2] = data[0] * other.data[1] - data[1] * other.data[0];
        return cudVec3D_f(result);
    }

    // Dot product
    __device__ inline float dot(const cudVec3D_f &other)
    {
        float result = data[0] * other.data[0] + data[1] * other.data[1] + data[2] * other.data[2];
        return result;
    }

    // Sum reduction
    __device__ inline float sum()
    {
        float sum = data[0] + data[1] + data[2];
        return sum;
    }

    // Normalization
    __device__ inline cudVec3D_f normalize()
    {
        float result[3];
        float norm = sqrtf(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
        result[0] = data[0] / norm;
        result[1] = data[1] / norm;
        result[2] = data[2] / norm;
        return cudVec3D_f(result);
    }

    // Overloading +
    __device__ cudVec3D_f operator+(const cudVec3D_f &other)
    {
        return this->add(other);
    }

    // Overloading -
    __device__ cudVec3D_f operator-(const cudVec3D_f &other)
    {
        return this->sub(other);
    }

    // Overloading * for pointwise multiplication
    __device__ cudVec3D_f operator*(const cudVec3D_f &other){
        return this->prod(other);
    }

    // Overloading * for scalar multiplication
    __device__ cudVec3D_f operator*(const float &scalar){
        return this->prod(scalar);
    }

    // Overloading indexing
    __device__ float &operator[](int i)
    {
        float &elem = data[i];
        return elem;
    }
};

#endif