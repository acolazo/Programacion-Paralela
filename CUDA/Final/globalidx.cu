#include <stdio.h>
#include <time.h>
#include <math.h>

#define SIZE 150 * 1000
#define THREADS 256 //best value = 256

#define SORT 1
#define TestReduction 0
#define PRINT 0
#define printErrors 0
#define CHECK 1
#define DATATYPE int
#define RECORDTIME 1
#define MIN INT_MIN
#define PRINTINFO 0

#define OPTION 1
/*
1: i+1 
2: SIZE-i
3: rand() % 100
*/

#define CUDA_ERROR_CHECK

/* Function declarations */
void getGridComposition(int, unsigned int *, unsigned int *, unsigned int);
void printResults(DATATYPE *);


/* Error Checking */

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif
    return;
}

inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }

    /* Can affect performance. Comment if needed. */
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif
    return;
}

/* Kernel reduction at block level */
/* One thread per data */
/* Input list of indexes */
/* Outputs the index of the maximum value so we can later use that index to sort the list */
__global__ void reduceWithIndexKernel(int size, DATATYPE * g_list, int * g_inidx, int * g_outidx)
{
    unsigned int tid = threadIdx.x;
    unsigned int gid = (blockIdx.x * blockDim.x) + tid;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            g_inidx[gid] = g_list[g_inidx[gid]] > g_list[g_inidx[gid + s]] ? g_inidx[gid] : g_inidx[gid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_outidx[blockIdx.x] = g_inidx[gid];
}

/* Kernel reduction at block level */
/* One thread per data */
/* Input list of indexes */
/* Outputs the index of the maximum value so we can later use that index to sort the list */
__global__ void reduceKernel(int size, DATATYPE * g_list, int * g_inidx, int * g_outidx)
{
    unsigned int tid = threadIdx.x;
    unsigned int gid = (blockIdx.x * blockDim.x * 2) + tid;

    g_inidx[gid] = g_list[gid] > g_list[gid + blockDim.x] ? gid : gid + blockDim.x;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            g_inidx[gid] = g_list[g_inidx[gid]] > g_list[g_inidx[gid + s]] ? g_inidx[gid] : g_inidx[gid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_outidx[blockIdx.x] = g_inidx[gid];
}

/* This function swaps the MAX element [which should be in position 0] with the last element of the list */
/* Also it will save the max in determined position in g_sorted_list */
/* WARNING: This function must be called by only one block */
__global__ void swapKernel(int size, DATATYPE *g_list, DATATYPE *g_max, DATATYPE * g_sorted_list)
{
    int max_index;
    int max;

        max_index = g_max[0];
        /* First we swap elements and fill the maximum with the MIN value */
        max = g_list[max_index];
        g_list[max_index] = g_list[size - 1];
        g_list[size - 1] = MIN;
        /* Then we save the maximum in the last available position of the array */
        g_sorted_list[size - 1] = max;

}

/* Kernel call that wraps data into a struct with index */
__global__ void fillDataKernel(int size, DATATYPE * g_list)
{
    unsigned int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid >= size){
        g_list[gid] = MIN;
    }
}

/* Wraps Reduction Kernel Call */
DATATYPE * reduceMax(int size, DATATYPE *g_list, DATATYPE *g_wa, DATATYPE *g_wb)
{

    unsigned int threads, blocks;
    DATATYPE *input, *output, *ptr;
    int N, temp;
    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(1, 1, 1);

    N = size;
    getGridComposition(size, &blocks, &threads, 2);
    dimGrid.x = blocks;
    dimBlock.x = threads;

    if(PRINTINFO) printf("Bloques: %d, Threads: %d, N: %d\n", dimGrid.x, dimBlock.x, N);
    /* First Reduce */
    reduceKernel<<<dimGrid, dimBlock>>>(N, g_list, g_wa, g_wb); /* Loads 2 data per thread */
    CudaCheckError();
    temp = (N / (dimBlock.x * 2));
        if (N % (dimBlock.x * 2) != 0)
            temp++;
    N = temp;

    if(dimGrid.x == 1) {
        dimGrid.x = 0;
        input = g_wa;
    }
    else {
        getGridComposition(N, &blocks, &threads, 1);
        dimGrid.x = blocks;
        dimBlock.x = threads;
        input = g_wb;
        output = g_wa;
    }
    /* End of first reduce */
    while (dimGrid.x > 0)
    {
        if(PRINTINFO) printf("Bloques: %d, Threads: %d, N: %d\n", dimGrid.x, dimBlock.x, N);
        reduceWithIndexKernel<<<dimGrid, dimBlock>>>(N, g_list, input, output);
        CudaCheckError();

        temp = (N / dimBlock.x);
        if (N % (dimBlock.x) != 0)
            temp++;
        N = temp;
        
        ptr = input;
        input = output;
        output = ptr;

        if(dimGrid.x == 1) dimGrid.x = 0;
        else {
            getGridComposition(N, &blocks, &threads, 1);
            dimGrid.x = blocks;
            dimBlock.x = threads;
        }
    }

    return input; /* At this point input is the last output */
}

/* Calls the iterative reduction wrapper and sorts the max results */
void sortBySelectionIterative(int size, DATATYPE *g_list, DATATYPE *g_wa, DATATYPE *g_wb, DATATYPE * g_sorted_list)
{
    DATATYPE * max;
    
    for (int i = size; i > 0; i--)
    {
        max = reduceMax(i, g_list, g_wa, g_wb);
        swapKernel<<<1, 1>>>(i, g_list, max, g_sorted_list);

        /* Test */
        /*
        DATATYPE test_list[SIZE];
        CudaSafeCall(cudaMemcpy(test_list, g_list, SIZE * sizeof(DATATYPE), cudaMemcpyDeviceToHost));
        printResults(test_list);
        */
        
    }

    return;
}

/* Get the number of blocks and threads per block */
void getGridComposition(int size, unsigned int *blocks, unsigned int *threads, unsigned int data_per_thread)
{

    *threads = THREADS;
    *blocks = 1;

    while (((*blocks) * (*threads)) * data_per_thread < size)
    {
        *blocks <<= 1;
    }


    if(*blocks == 1){
        while( *threads >= size * 2 / data_per_thread )
            *threads >>= 1;

            if (*threads == 0) *threads = 1;
    }

    return;
}

void printResults(DATATYPE *sorted_list)
{
    if (printErrors)
    {
        for (int i = 0; i < SIZE; i++)
        {
            if (sorted_list[i] != (i + 1))
                printf("%d: %d \n", i + 1, sorted_list[i]);
        }
        printf("\n");
    }
    else
    {
        for (int i = 0; i < SIZE; i++)
        {
            printf("%d\n", sorted_list[i]);
        }
        printf("\n");
    }

    return;
}

int checkResults(DATATYPE *sorted_list)
{
    unsigned int check = 1;
    unsigned int i;
    for (i = 0; i < SIZE; i++)
    {
        if (sorted_list[i] != (i + 1))
            check = 0;
    }

    if (check)
        printf("Resultados correctos!\n");
    else
        printf("Resultados incorrectos!\n");

    return check;
}

int main(void)
{
    DATATYPE *g_list, *g_sorted_list, *g_wa, *g_wb;
    DATATYPE * list;
    srand(time(NULL));
    /* Allocate Host memory */
    list = (DATATYPE *)malloc(SIZE * sizeof(DATATYPE));
    if (list == NULL)
    {
        printf("Error alocando memoria.\n");
        exit(-1);
    }

    /* Allocate device memory */
    

    unsigned int allocate = 2;
    while( allocate < SIZE )
        allocate <<= 1;

    printf("Allocated memory: %d\n", allocate);
    CudaSafeCall(cudaMalloc((void **)&g_sorted_list, SIZE * sizeof(DATATYPE)));
    CudaSafeCall(cudaMalloc((void **)&g_list, allocate * sizeof(DATATYPE)));
    CudaSafeCall( cudaMalloc((void**)&g_wa, allocate * sizeof(DATATYPE) ) );
    //CudaSafeCall( cudaMalloc((void**)&g_wb, (allocate / THREADS)  * sizeof(DATATYPE) ) );
    CudaSafeCall( cudaMalloc((void**)&g_wb, ((allocate / THREADS) + 1)  * sizeof(DATATYPE) ) );
    

    /* Initialize data */
    for (int i = 0; i < SIZE; i++)
    {
        switch (OPTION)
        {
        case 1:
            list[i] = i + 1;
            break;
        case 2:
            list[i] = SIZE - i;
            break;
        case 3:
            list[i] = rand() % 100;
            break;
        }        
    }

    /* Wrap Data into a struct with index for sorting */
    unsigned int threads, blocks;
    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(1, 1, 1);

    CudaSafeCall( cudaMemcpy(g_list, list, SIZE * sizeof(DATATYPE), cudaMemcpyHostToDevice) );


    /* Pad data */
    getGridComposition(allocate, &blocks, &threads, 1);
    dimGrid.x = blocks;
    dimBlock.x = threads;
    if(PRINTINFO) printf("Bloques: %d, Threads: %d, N: %d\n", dimGrid.x, dimBlock.x, SIZE);

    fillDataKernel<<<dimGrid, dimBlock>>>(SIZE, g_list);
    CudaCheckError();
    /* End of padding data */


    if (SORT){
        /* Record time */
        cudaEvent_t start, stop;
        if (RECORDTIME)
        {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
        }
        sortBySelectionIterative(SIZE, g_list, g_wa, g_wb, g_sorted_list);
        if (RECORDTIME)
        {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("Pasaron %f milisegundos\n", milliseconds);
        }

        CudaSafeCall(cudaMemcpy(list, g_sorted_list, SIZE * sizeof(DATATYPE), cudaMemcpyDeviceToHost));
    } 

    DATATYPE * result;
    DATATYPE maxidx;
    if (TestReduction){
        /* Record time */
        cudaEvent_t start, stop;
        if (RECORDTIME)
        {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
        }

        result = reduceMax(SIZE, g_list, g_wa, g_wb);

        if (RECORDTIME)
        {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("Pasaron %f milisegundos\n", milliseconds);
        }

        CudaSafeCall(cudaMemcpy(&maxidx, result, 1 * sizeof(DATATYPE), cudaMemcpyDeviceToHost));
    } 
    

    if(TestReduction){
        printf("The max index is: %d\n", maxidx);
        printf("El maximo es %d\n", list[maxidx]);
    }

    if (PRINT && SORT)
    {
        printResults(list);
    }
    if(CHECK && SORT)
    {
        checkResults(list);
    }

    CudaSafeCall ( cudaFree(g_list) );
    CudaSafeCall ( cudaFree(g_wa) );
    CudaSafeCall ( cudaFree(g_wb) );
    CudaSafeCall ( cudaFree(g_sorted_list) );

    free(list);
}