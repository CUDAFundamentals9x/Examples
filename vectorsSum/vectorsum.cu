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
//---------------------------------------------------------------------------
void hostAllocation(float *&hVecA, float *&hVecB, float *&hVecC, size_t size)
{
    // Allocate the host input vectors
    hVecA = (float*)malloc(size);
    hVecB = (float*)malloc(size);
    hVecC = (float*)malloc(size);

    // Verify allocations
    if (hVecA == NULL || hVecB == NULL || hVecC == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
}

//---------------------------------------------------------------------------
//  hostInitialization
//  Initialize host input vectors
//---------------------------------------------------------------------------
void hostInitialization(float *&hVecA, float *&hVecB, int N)
{
    for (int i = 0; i < N; ++i)
    {
        hVecA[i] = rand()/(float)RAND_MAX;
        hVecB[i] = rand()/(float)RAND_MAX;
    }
}

//---------------------------------------------------------------------------
//  deviceAllocation
//  Allocate device output vectors
//---------------------------------------------------------------------------
void deviceAllocation(float *&dVecA, float *&dVecB, float *&dVecC, size_t size)
{
    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**)&dVecA, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&dVecB, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&dVecC, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

//---------------------------------------------------------------------------
//  copyHostToDevice
//  Copy host input vectors to device input vectors in GPU memory
//---------------------------------------------------------------------------
void copyHostToDevice(float *&dVecA, float *&dVecB, float *&hVecA, float *&hVecB, size_t size)
{
    cudaError_t err = cudaSuccess;

    printf("Copy input data from CPU memory to CUDA GPU device\n");
    err = cudaMemcpy(dVecA, hVecA, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(dVecB, hVecB, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

//---------------------------------------------------------------------------
//  copyDeviceToHost
//  Copy device result vector (GPU memory) to host result vector (CPU memory)
//---------------------------------------------------------------------------
void copyDeviceToHost(float *&hVecC, float *&dVecC, size_t size)
{
    cudaError_t err = cudaSuccess;

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(hVecC, dVecC, size, cudaMemcpyDeviceToHost);

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
void verifyResultVectors(float *hVecA, float *hVecB, float *hVecC, int N)
{
    for (int i = 0; i < N; ++i)
    {
        if (fabs(hVecA[i] + hVecB[i] - hVecC[i]) > 1e-5)
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
void freeDeviceMemory(float *&dVecA, float *&dVecB, float *&dVecC)
{
    cudaError_t err = cudaSuccess;

    err = cudaFree(dVecA);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(dVecB);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(dVecC);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

//---------------------------------------------------------------------------
//  freeHostMemory
//  Free host memory
//---------------------------------------------------------------------------
void freeHostMemory(float *&hVecA, float *&hVecB, float *&hVecC)
{
    free(hVecA);
    free(hVecB);
    free(hVecC);
}

//---------------------------------------------------------------------------
//  Host vectorSum function
//---------------------------------------------------------------------------
int vectorSum(int numElements)
{
    cudaError_t err = cudaSuccess;

    // Host vectors
    float *hVecA = NULL, *hVecB = NULL, *hVecC = NULL;

    // Device vectors
    float *dVecA = NULL, *dVecB = NULL, *dVecC = NULL;

    // Print the vector length
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Host memory allocation
    hostAllocation(hVecA, hVecB, hVecC, size);

    // Host vectors inicialization
    hostInitialization(hVecA, hVecB, numElements);

    // Device memory allocation
    deviceAllocation(dVecA, dVecB, dVecC, size);

    // Copy host input vectors (CPU memory) to device input vectors (GPU memory)
    copyHostToDevice(dVecA, dVecB, hVecA, hVecB, size);

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

    // Verify that the result vector is correct
    verifyResultVectors(hVecA, hVecB, hVecC, numElements);

    // Free device global memory
    freeDeviceMemory(dVecA, dVecB, dVecC);

    // Free host memory
    freeHostMemory(hVecA, hVecB, hVecC);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state
    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

