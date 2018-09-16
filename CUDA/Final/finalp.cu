#include <stdio.h>
#include <time.h>
#include <math.h>

#define SIZE 8 * 1000
#define maxSharedMemory 49152 //bytes
#define THREADS 256 //best value = 256

#define SORT 0
#define TestReduction 1
#define PRINT 0
#define printErrors 0
#define CHECK 1

#define DATATYPE struct number
#define VALUETYPE int
#define MINVALUE INT_MIN

#define OPTION 1
/*
1: i+1 
2: SIZE-i
3: rand() % 100
*/

#define RECORDTIME 1
//#define CUDA_ERROR_CHECK

/* Function declarations */
void getGridComposition(int, unsigned int*, unsigned int*, unsigned int);

/* Struct for not losing the global index */
struct number{
    VALUETYPE value;
    int index;
};

/* Error Checking */


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

/* Warp reduction */
template <unsigned int blockSize>
__device__ void warpReduce(DATATYPE* sdata, unsigned int tid, unsigned int i, int size){
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
    unsigned int i = (blockIdx.x*blockDim.x * 2) + tid;
    
    if((i + blockDim.x )< size)
        sdata[tid] = g_input[i].value > g_input[i + blockDim.x].value ? g_input[i] : g_input[ i + blockDim.x];  
    else if (i<size){
        sdata[tid] = g_input[i];
    }
    
    else{
        DATATYPE min_value;
        min_value.value = MINVALUE;
        sdata[tid] = min_value;
    }
    
    
        

    __syncthreads();

    /* Unrolling all iterations */
    if (blockSize >= 1024) {
        if (tid < 512) {
            sdata[tid] = sdata[tid].value > sdata[tid + 512].value ? sdata[tid] : sdata[tid + 512]; 
        }
        __syncthreads(); 
    }
    if (blockSize >= 512) {
        if (tid < 256) { 
            sdata[tid] = sdata[tid].value > sdata[tid + 256].value ? sdata[tid] : sdata[tid + 256];
        }
        __syncthreads(); 
    }
    if (blockSize >= 256) {
        if (tid < 128) { 
            sdata[tid] = sdata[tid].value > sdata[tid + 128].value ? sdata[tid] : sdata[tid + 128];
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) {
        if (tid <  64) { 
            sdata[tid] = sdata[tid].value > sdata[tid + 64].value ? sdata[tid] : sdata[tid + 64];
        }
        __syncthreads(); 
    }   
    
    if(tid < 32){
        warpReduce<blockSize>(sdata, tid, i, size);
    }

    //write result for this block to global mem
    if (tid == 0) g_output[blockIdx.x] = sdata[tid];   
}

/* This function swaps the MAX element [which should be in position 0] with the last element of the list */
/* WARNING: This function must be called by only one block */
__global__ void swapKernel(int size, DATATYPE *g_list, DATATYPE *g_max, VALUETYPE *g_output){
    DATATYPE max;
    DATATYPE last_element;
    int index;
    //unsigned int tid = threadIdx.x;

        max = g_max[0];
        index = max.index;
        max.index = size-1;

        last_element = g_list[size-1];
        last_element.index = index;

        g_list[index] = last_element; /* Donde estaba el valor maximo, pongo el ultimo elemento de la lista */
        
        g_output[size - 1] = max.value; /* Pongo el maximo en la lista de resultados */

        max.value = MINVALUE; /* Cambio el valor del maximo para que sea el minimo posible */
        g_list[size-1] = max;

}


/* Kernel call that wraps data into a struct with index */
__global__ void wrapKernel(int size, DATATYPE *g_wrapped_list, VALUETYPE * g_list ){
    unsigned int gid = threadIdx.x + blockDim.x * blockIdx.x;
    DATATYPE myData;

    if ( gid < size ){
        myData.index = gid;
        myData.value = g_list[gid];

        g_wrapped_list[gid] = myData;
    }
}

/* Kernel call that unwraps data into an array of VALUETYPE */
__global__ void unwrapKernel(int size, DATATYPE *g_wrapped_list, VALUETYPE * g_list ){
    unsigned int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if ( gid < size ){
         g_list[gid] = g_wrapped_list[gid].value;
    }
}


/* Wraps Reduction Kernel Call */
int reduceMax(int size, DATATYPE *g_list, DATATYPE *g_temp, DATATYPE *g_temp_results){
    static int counter = 0;
    unsigned int threads, blocks;
    DATATYPE *g_input, *g_output, *g_iteration_list;
    int N, iterations, maxAllowedSize, CONST_N;
    unsigned int maxLoadToShared, A;
    double temp;
    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(1, 1, 1);
    

    g_iteration_list = g_list;
    /* Check if we can do one kernel call or more */
    iterations = 1;
    CONST_N = size;

    maxLoadToShared = 1;
    while(maxLoadToShared * 2 * sizeof(DATATYPE) < maxSharedMemory)
        maxLoadToShared<<=1;



    maxAllowedSize = maxLoadToShared * 2;

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
            /* */
            A = dimGrid.x * dimBlock.x;
            /* */

            if(dimGrid.x == 1){
                g_output = g_temp_results + i;
            }
            counter++;
            switch(dimBlock.x){
                case 1024:
                    reduceKernel<1024><<<dimGrid, dimBlock, A * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 512:
                    reduceKernel<512><<<dimGrid, dimBlock, A * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 256:
                    reduceKernel<256><<<dimGrid, dimBlock, A * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 128:
                    reduceKernel<128><<<dimGrid, dimBlock, A * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 64:
                    reduceKernel<64><<<dimGrid, dimBlock, A * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 32:
                    reduceKernel<32><<<dimGrid, dimBlock, A * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;  
                case 16:
                    reduceKernel<16><<<dimGrid, dimBlock, A * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 8:
                    reduceKernel<8><<<dimGrid, dimBlock, A * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 4:
                    reduceKernel<4><<<dimGrid, dimBlock, A * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 2:
                    reduceKernel<2><<<dimGrid, dimBlock, A * sizeof(DATATYPE)>>>(N, g_input, g_output);
                    break;
                case 1:
                    reduceKernel<1><<<dimGrid, dimBlock, A * sizeof(DATATYPE)>>>(N, g_input, g_output);
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
    

    return counter;
}


/* Calls the reduction wrapper and sorts the max results */
void sortBySelection(int size, DATATYPE *g_list, DATATYPE *g_temp, DATATYPE * g_temp_results, VALUETYPE * g_results){
    reduceMax(size, g_list, g_temp, g_temp_results);

    swapKernel<<<1, 1>>>(size, g_list, g_temp_results, g_results);

    if(size > 2){
        sortBySelection(size-1, g_list, g_temp, g_temp_results, g_results);
    }
    

    return;
}

/* Calls the iterative reduction wrapper and sorts the max results */
void sortBySelectionIterative(int size, DATATYPE *g_list, DATATYPE *g_temp, DATATYPE * g_temp_results, VALUETYPE * g_results){

    for(int i = size; i > 0; i--){
        reduceMax(i, g_list, g_temp, g_temp_results);
        swapKernel<<<1, 1>>>(i, g_list, g_temp_results, g_results);
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
        while( ( *threads * data_per_thread / 2 ) > size && (*threads > 1)){
            *threads >>=1;
        }
    }
    */
    

    return;
}

void printResults(VALUETYPE *sorted_list)
{   
    if(printErrors){
        for (int i = 0; i < SIZE; i++)
        {
            if(sorted_list[i] != (i + 1)) printf("%d: %d \n", i + 1, sorted_list[i]);
        }
        printf("\n");
    }
    else{
        for (int i = 0; i < SIZE; i++)
        {
            printf("%d\n", sorted_list[i]);
        }
        printf("\n");
    }
    
    return;
}

int checkResults(VALUETYPE *sorted_list){
    unsigned int check = 1;
    unsigned int i;
    for (i = 0; i < SIZE; i++)
    {
        if(sorted_list[i] != (i + 1))
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
    DATATYPE *list_g, *list_g_o, *g_temp_results;
    VALUETYPE *list, *list_g_unwrapped;
    int allocate_exceded_share_mem;
    srand(time(NULL));

    /* Allocate Host memory */
    list = (VALUETYPE *)malloc(SIZE * sizeof(VALUETYPE));
    if(list == NULL){
        printf("Error alocando memoria.\n");
        exit(-1);
    }

    /* Allocate device memory */
    CudaSafeCall( cudaMalloc((void**)&list_g_unwrapped, SIZE * sizeof(VALUETYPE)) );
    CudaSafeCall( cudaMalloc((void**)&list_g, SIZE * sizeof(DATATYPE)) );
    CudaSafeCall( cudaMalloc((void**)&list_g_o, maxSharedMemory / THREADS ) );

    allocate_exceded_share_mem = 1;
    for(int i = (SIZE * sizeof(DATATYPE)); i > maxSharedMemory; i-=maxSharedMemory)
        allocate_exceded_share_mem++;

    CudaSafeCall (cudaMalloc((void**)&g_temp_results, allocate_exceded_share_mem * sizeof(DATATYPE)) );

    /* Initialize data */
   /* Initialize data */
   for (int i = 0; i < SIZE; i++)
   {   
       switch(OPTION){
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

    CudaSafeCall( cudaMemcpy(list_g_unwrapped, list, SIZE * sizeof(VALUETYPE), cudaMemcpyHostToDevice) );

    getGridComposition(SIZE, &blocks, &threads, 1);
    dimGrid.x = blocks;
    dimBlock.x = threads;

    wrapKernel<<<dimGrid, dimBlock>>>(SIZE, list_g, list_g_unwrapped );
    /* End of wrapping data */


    if (SORT){
        /* Record time */
        cudaEvent_t start, stop;
        if(RECORDTIME){
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
        }
        sortBySelectionIterative(SIZE, list_g, list_g_o, g_temp_results, list_g_unwrapped);
        if(RECORDTIME){
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("Pasaron %f milisegundos\n", milliseconds);
        }
        CudaSafeCall( cudaMemcpy(list,list_g_unwrapped , SIZE * sizeof(VALUETYPE), cudaMemcpyDeviceToHost) );
    } 

    if (TestReduction){
        /* Record time */
        int contador;
        cudaEvent_t start, stop;
        if(RECORDTIME){
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
        }
        contador = reduceMax(SIZE, list_g, list_g_o, g_temp_results);
        if(RECORDTIME){
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("Pasaron %f milisegundos\n", milliseconds);
        } 
        unwrapKernel<<<1, 1>>>(1, g_temp_results, list_g_unwrapped );
        printf("Entro %d veces a la funcion de kernel\n", contador);
        CudaSafeCall( cudaMemcpy(list,list_g_unwrapped , 1 * sizeof(VALUETYPE), cudaMemcpyDeviceToHost) );

    } 
    
    /* Unwrap data into an array of VALUETYPE */
    // unwrapKernel<<<dimGrid, dimBlock>>>(SIZE, list_g, list_g_unwrapped );

    /* End of unwrapping data */

    if(TestReduction){
        printf("El maximo es %d\n", list[0]);
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
    CudaSafeCall ( cudaFree(list_g_unwrapped) );
    free(list);
}