//---------------------------------------------------------------------------
//  Example of a simple vector sum on the GPU and on the host.
//  CUDA parameters convention:
//      function(dst, src)
//    	dst 	- Destination memory address
//    	src 	- Source memory address
//---------------------------------------------------------------------------
#include <stdio.h>
#include <cuda_runtime.h>

//---------------------------------------------------------------------------
//  GPU vectorSum
//  CUDA kernel to sum elements of vectors A and B and returns vector C
//---------------------------------------------------------------------------
__global__ void
vectorSum(const float *A, const float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

//---------------------------------------------------------------------------
//  hostAllocation
//  Allocate the host input vector (CPU memory)
//---------------------------------------------------------------------------
float* hostAllocation(size_t size)
{
    float* hostVector = (float*)malloc(size);

    // Verify allocations
    if (hostVector == nullptr)
    {
        fprintf(stderr, "Failed to allocate host vector!\n");
        exit(EXIT_FAILURE);
    }

    return hostVector;
}

//---------------------------------------------------------------------------
//  hostInitialization
//  Initialize host input vector
//---------------------------------------------------------------------------
void hostInitialization(float* hostVector, int N)
{
    for (int i = 0; i < N; ++i)
    {
        hostVector[i] = rand()/(float)RAND_MAX;
    }
}

//---------------------------------------------------------------------------
//  deviceAllocation
//  Allocate device output vector (GPU memory)
//---------------------------------------------------------------------------
float* deviceAllocation(size_t size)
{
    cudaError_t err = cudaSuccess;

	float* deviceVector;
	
    err = cudaMalloc((void**)&deviceVector, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return deviceVector;
}

//---------------------------------------------------------------------------
//  copyHostToDevice
//  Copy host input vector (CPU memory) to device input vector (GPU memory)
//---------------------------------------------------------------------------
void copyHostToDevice(float* deviceVector, float* hostVector, size_t size)
{
    cudaError_t err = cudaSuccess;

    printf("Copy input data from CPU memory to GPU memory (CUDA device)\n");
    err = cudaMemcpy(deviceVector, hostVector, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

//---------------------------------------------------------------------------
//  copyDeviceToHost
//  Copy device vector (GPU memory) to host vector (CPU memory)
//---------------------------------------------------------------------------
void copyDeviceToHost(float* hostVector, float* deviceVector, size_t size)
{
    cudaError_t err = cudaSuccess;

    printf("Copy data from the GPU memory (CUDA device) to the CPU memory (host)\n");
    err = cudaMemcpy(hostVector, deviceVector, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

//---------------------------------------------------------------------------
//  verifyResultVectors
//  Verify that the result vector is correct
//---------------------------------------------------------------------------
void verifyResultVectors(float* hostVectorA, float* hostVectorB, float* hostVectorC, int N)
{
    for (int i = 0; i < N; ++i)
    {
        if (fabs(hostVectorA[i] + hostVectorB[i] - hostVectorC[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
}

//---------------------------------------------------------------------------
//  freeDeviceMemory
//  Free device global memory
//---------------------------------------------------------------------------
void freeDeviceMemory(float* &deviceVector)
{
    cudaError_t err = cudaSuccess;

    err = cudaFree(deviceVector);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

//---------------------------------------------------------------------------
//  Host vectorSum function
//---------------------------------------------------------------------------
int vectorSum(int numElements)
{
    cudaError_t err = cudaSuccess;

   
    // Print the vector length
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Host memory allocation
    float* hVecA = hostAllocation(size);
    float* hVecB = hostAllocation(size);
    float* hVecC = hostAllocation(size);

    // Host vectors inicialization
    hostInitialization(hVecA, numElements);
    hostInitialization(hVecB, numElements);

    // Device memory allocation
    float* dVecA = deviceAllocation(size);
    float* dVecB = deviceAllocation(size);
    float* dVecC = deviceAllocation(size);

    // Copy host input vectors (CPU memory) to device input vectors (GPU memory)
    copyHostToDevice(dVecA, hVecA, size);
    copyHostToDevice(dVecB, hVecB, size);

    // Launch the CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vectorSum<<<blocksPerGrid, threadsPerBlock>>>(dVecA, dVecB, dVecC, numElements);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy device resutl vector to host result vector
    copyDeviceToHost(hVecC, dVecC, size);

    // Verify that the host result vectors are correct
    verifyResultVectors(hVecA, hVecB, hVecC, numElements);

    // Free device global memory
    freeDeviceMemory(dVecA);
    freeDeviceMemory(dVecB);
    freeDeviceMemory(dVecC);

    // Free host memory
    free(hVecA);
    free(hVecB);
    free(hVecC);

    // Reset the device and exit
    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

