#include <stdio.h>
#include <time.h>
#include <math.h>

#define SIZE 100 * 1000
#define maxSharedMemory 49152 //bytes
#define THREADS 256 //best value = 256
#define SORT 1
#define TestReduction 0
#define PRINT 0
#define printErrors 0
#define CHECK 1
#define DATATYPE struct number

/* Error Checking */
//#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    #ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err ){
        fprintf ( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err) );
        exit(-1);

    }
    #endif
    return;
}

inline void __cudaCheckError ( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if(cudaSuccess != err){
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
        exit(-1);
    }

    /* Can affect performance. Comment if needed. */
    err = cudaDeviceSynchronize();
    if(cudaSuccess != err){
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
        exit(-1);
    }
    #endif
    return;
}

/* Function declarations */
void getGridComposition(int, unsigned int*, unsigned int*);

/* Struct for not losing the global index */
struct number{
    int value;
    unsigned int index;
};

/* Warp reduction */
template <unsigned int blockSize>
__device__ void warpReduce(DATATYPE* sdata, unsigned int tid, unsigned int i, int size){
    if (blockSize >=64) if ((i + 32) < size) sdata[tid] = sdata[tid].value > sdata[tid + 32].value ? sdata[tid] : sdata[tid + 32];
    if (blockSize >=32) if ((i + 16) < size) sdata[tid] = sdata[tid].value > sdata[tid + 16].value ? sdata[tid] : sdata[tid + 16];
    if (blockSize >=16) if ((i + 8) < size)  sdata[tid] = sdata[tid].value > sdata[tid + 8].value ? sdata[tid] : sdata[tid + 8];
    if (blockSize >=8) if ((i + 4) < size)  sdata[tid] = sdata[tid].value > sdata[tid + 4].value ? sdata[tid] : sdata[tid + 4];
    if (blockSize >=4) if ((i + 2) < size)  sdata[tid] = sdata[tid].value > sdata[tid + 2].value ? sdata[tid] : sdata[tid + 2];
    if (blockSize >=2) if ((i + 1) < size)  sdata[tid] = sdata[tid].value > sdata[tid + 1].value ? sdata[tid] : sdata[tid + 1];
};

/* Warp Performant Reduce */
template <unsigned int blockSize>
__device__ void warpPerformantReduce(DATATYPE* sdata, unsigned int tid, unsigned int i, int size){
    if (blockSize >=64) sdata[tid] = sdata[tid].value > sdata[tid + 32].value ? sdata[tid] : sdata[tid + 32];
    if (blockSize >=32) sdata[tid] = sdata[tid].value > sdata[tid + 16].value ? sdata[tid] : sdata[tid + 16];
    if (blockSize >=16) sdata[tid] = sdata[tid].value > sdata[tid + 8].value ? sdata[tid] : sdata[tid + 8];
    if (blockSize >=8)  sdata[tid] = sdata[tid].value > sdata[tid + 4].value ? sdata[tid] : sdata[tid + 4];
    if (blockSize >=4)  sdata[tid] = sdata[tid].value > sdata[tid + 2].value ? sdata[tid] : sdata[tid + 2];
    if (blockSize >=2)  sdata[tid] = sdata[tid].value > sdata[tid + 1].value ? sdata[tid] : sdata[tid + 1];
};

/* Kernel reduction at block level */
template <unsigned int blockSize>
__global__ void reduceKernel(int size, DATATYPE *g_input, DATATYPE *g_output)
{
    extern __shared__ DATATYPE sdata[];

    unsigned int tid = threadIdx.x;
  //  unsigned int gid = (blockIdx.x*blockDim.x) + tid;
    unsigned int i = (blockIdx.x*blockDim.x * 2) + tid;
    if((i + blockDim.x )< size)
        sdata[tid] = g_input[i].value > g_input[i + blockDim.x].value ? g_input[i] : g_input[ i + blockDim.x];  
    else if (i<size)
        sdata[tid] = g_input[i];

    __syncthreads();

    /* Unrolling all iterations */
    if (blockSize >= 1024) {
        if (tid < 512) {
            if ((i + 512) < size) sdata[tid] = sdata[tid].value > sdata[tid + 512].value ? sdata[tid] : sdata[tid + 512]; 
        }
        __syncthreads(); 
    }
    if (blockSize >= 512) {
        if (tid < 256) { 
            if ((i + 256) < size) sdata[tid] = sdata[tid].value > sdata[tid + 256].value ? sdata[tid] : sdata[tid + 256];
        }
        __syncthreads(); 
    }
    if (blockSize >= 256) {
        if (tid < 128) { 
            if ((i + 128) < size) sdata[tid] = sdata[tid].value > sdata[tid + 128].value ? sdata[tid] : sdata[tid + 128];
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) {
        if (tid <  64) { 
            if ((i + 64) < size) sdata[tid] = sdata[tid].value > sdata[tid + 64].value ? sdata[tid] : sdata[tid + 64];
        }
        __syncthreads(); 
    }   
    
    if(tid < 32){
        if( (i + 64) < size ) warpPerformantReduce<blockSize>(sdata, tid, i, size);
        else warpReduce<blockSize>(sdata, tid, i, size);
    }

    //write result for this block to global mem
    if (tid == 0) g_output[blockIdx.x] = sdata[tid];   
}

/* This function swaps the MAX element [which should be in position 0] with the last element of the list */
/* WARNING: This function must be called by only one block */
__global__ void swapKernel(int size, DATATYPE *g_list, DATATYPE *g_max){
    DATATYPE max;
    DATATYPE last_element;
    int index;
    unsigned int tid = threadIdx.x;

    if(tid == 0){
        max = g_max[0];
        index = max.index;
        max.index = size-1;

        last_element = g_list[size-1];
        last_element.index = index;

        g_list[index] = last_element; /* Donde estaba el valor maximo, pongo el ultimo elemento de la lista */
        g_list[size-1] = max;
    }
}

/* Wraps Kernel Calls */
void reduceMax(int size, DATATYPE *g_list, DATATYPE *g_temp, DATATYPE *g_temp_results){
    unsigned int threads, blocks;
    DATATYPE *g_input, *g_output, *g_iteration_list;
    int N, iterations, maxAllowedSize, CONST_N;
    double temp;
    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(1, 1, 1);
    

    g_iteration_list = g_list;
    /* Check if we can do one kernel call or more */
    iterations = 1;
    CONST_N = size;
    maxAllowedSize = maxSharedMemory / sizeof(DATATYPE);

    while(CONST_N > maxAllowedSize ){
       CONST_N -= maxAllowedSize;
       iterations++;
    }
    
    for(int i = 0; i < iterations; i++){
        /* Get Grid Composition */
        g_input = g_iteration_list;
        N = CONST_N;
        getGridComposition(N, &blocks, &threads, 2);
        dimGrid.x = blocks;
        dimBlock.x = threads;

        //printf("N: %d, Blocks: %d, Threads: %d\n", N, blocks, threads);

        g_output = g_temp;
        /* Perform the reduction for N elements */
        while(dimGrid.x > 0){

            if(dimGrid.x == 1){
                g_output = g_temp_results + i;
            }

            switch(dimBlock.x){
                case 1024:
                    reduceKernel<1024><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 512:
                    reduceKernel<512><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 256:
                    reduceKernel<256><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 128:
                    reduceKernel<128><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 64:
                    reduceKernel<64><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 32:
                    reduceKernel<32><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;  
                case 16:
                    reduceKernel<16><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 8:
                    reduceKernel<8><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 4:
                    reduceKernel<4><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 2:
                    reduceKernel<2><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 1:
                    reduceKernel<1><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
            }
                
            CudaCheckError();

            temp = (double) N / (dimBlock.x * 2);
            if (N % (dimBlock.x * 2) != 0) temp++;
            N = temp;

            dimGrid.x = (dimGrid.x > (dimBlock.x * 2)) || (dimGrid.x == 1) ? (dimGrid.x / (dimBlock.x * 2)) : 1;

            g_input = g_temp;
        }

        g_iteration_list += CONST_N;
        CONST_N = maxAllowedSize;
    }

   
    /* Recursive call to reduce Wrapper */
    if(iterations > 1){
        reduceMax(iterations, g_temp_results, g_temp, g_temp_results);
    }
    

    return;
}


/* Calls the reduction wrapper and sorts the max results */
void sortBySelection(int size, DATATYPE *g_list, DATATYPE *g_temp, DATATYPE * g_temp_results){

    reduceMax(size, g_list, g_temp, g_temp_results);

    swapKernel<<<1, 1>>>(size, g_list, g_temp_results);

    if(size > 2){
        sortBySelection(size-1, g_list, g_temp, g_temp_results);
    }
    

    return;
}

/* Calls the iterative reduction wrapper and sorts the max results */
void sortBySelectionIterative(int size, DATATYPE *g_list, DATATYPE *g_temp, DATATYPE * g_temp_results){
    for(int i = size; i > 1; i--){
        reduceMax(i, g_list, g_temp, g_temp_results);
        swapKernel<<<1, 1>>>(i, g_list, g_temp_results);
    }

    

    return;
}

/* Get the number of blocks and threads per block */
void getGridComposition(int size, unsigned int* blocks, unsigned int* threads, unsigned int data_per_thread){

    *threads = THREADS;
    *blocks = 1;

    while (((*blocks) * (*threads) * data_per_thread) < size){
        *blocks<<=1;
    }

    /*
    if(*blocks == 1){
        while( ( *threads * 2 ) > size && (*threads > 2)){
            *threads >>=1;
        }
    }
    */

    return;
}

void printResults(DATATYPE *sorted_list)
{   
    if(printErrors){
        for (int i = 0; i < SIZE; i++)
        {
            if(sorted_list[i].value != (i + 1)) printf("%d: %d (%d)\n", i + 1, sorted_list[i].value, sorted_list[i].index);
        }
        printf("\n");
    }
    else{
        for (int i = 0; i < SIZE; i++)
        {
            printf("%d\n", sorted_list[i].value);
        }
        printf("\n");
    }
    
    return;
}

int checkResults(DATATYPE *sorted_list){
    unsigned int check = 1;
    unsigned int i;
    for (i = 0; i < SIZE; i++)
    {
        if(sorted_list[i].value != (i + 1))
            check = 0;
    }

    if(check)
        printf("Resultados correctos!\n");
    else
        printf("Resultados incorrectos!\n");

    return check;
    
}

int main(void)
{
    DATATYPE *list, *list_g, *list_g_o, *g_temp_results;
    int allocate_exceded_share_mem;
    srand(time(NULL));

    list = (DATATYPE *)malloc(SIZE * sizeof(DATATYPE));
    if(list == NULL){
        printf("Error alocando memoria.\n");
        exit(0);
    }

    CudaSafeCall( cudaMalloc((void**)&list_g, SIZE * sizeof(DATATYPE)) );
    CudaSafeCall( cudaMalloc((void**)&list_g_o, maxSharedMemory / THREADS ) );


    allocate_exceded_share_mem = 1;
    for(int i = (SIZE * sizeof(DATATYPE)); i > maxSharedMemory; i-=maxSharedMemory)
        allocate_exceded_share_mem++;

    CudaSafeCall (cudaMalloc((void**)&g_temp_results, allocate_exceded_share_mem * sizeof(DATATYPE)) );

    for (int i = 0; i < SIZE; i++)
    {
        //list[i] = rand() % 100;
        list[i].value = SIZE - i;
        //list[i].value = i + 1;
        list[i].index = i;
    }

    CudaSafeCall( cudaMemcpy(list_g, list, SIZE * sizeof(DATATYPE), cudaMemcpyHostToDevice) );

    
    if (SORT){
        sortBySelectionIterative(SIZE, list_g, list_g_o, g_temp_results); /* Segmentation Fault */
        CudaSafeCall( cudaMemcpy(list,list_g , SIZE * sizeof(DATATYPE), cudaMemcpyDeviceToHost) );
    } 

    if (TestReduction){
        reduceMax(SIZE, list_g, list_g_o, g_temp_results);
        CudaSafeCall( cudaMemcpy(list,g_temp_results , sizeof(DATATYPE), cudaMemcpyDeviceToHost) );
        printf("El maximo es %d\n", list[0]);

        CudaSafeCall( cudaMemcpy(list,list_g , sizeof(DATATYPE), cudaMemcpyDeviceToHost) );
        printf("El ultimo de la lista es %d\n", list[SIZE - 1]);
    } 
    

    

    if (PRINT && SORT)
    {
        printResults(list);
    }
    if(CHECK && SORT)
    {
        checkResults(list);
    }

    printf("Allocated Exceded Mem: %d\n", allocate_exceded_share_mem);

    CudaSafeCall ( cudaFree(g_temp_results) );
    CudaSafeCall ( cudaFree(list_g) );
    CudaSafeCall ( cudaFree(list_g_o) );
    free(list);
}