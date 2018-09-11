#include <stdio.h>
#include <time.h>
#include <math.h>

#define SIZE 10000
#define THREADS 256 //best value = 256
#define PRINT 1
#define CHECK 1
#define DATATYPE struct number

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
    
    if(tid < 32) warpReduce<blockSize>(sdata, tid, i, size);

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
void sortBySelection(int size, DATATYPE *g_list, DATATYPE *g_temp){
    unsigned int threads, blocks;
    DATATYPE *g_input;
    int N;
    double temp;
    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(1, 1, 1);

    N = size;
    getGridComposition(N, &blocks, &threads);
    dimGrid.x = blocks;
    dimBlock.x = threads;

    g_input = g_list;

    while(dimGrid.x > 0){
        switch(dimBlock.x){
            case 1024:
                reduceKernel<1024><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_temp);
                break;
            case 512:
                reduceKernel<512><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_temp);
                break;
            case 256:
                reduceKernel<256><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_temp);
                break;
            case 128:
                reduceKernel<128><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_temp);
                break;
            case 64:
                reduceKernel<64><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_temp);
                break;
            case 32:
                reduceKernel<32><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_temp);
                break;  
            case 16:
                reduceKernel<16><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_temp);
                break;
            case 8:
                reduceKernel<8><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_temp);
                break;
            case 4:
                reduceKernel<4><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_temp);
                break;
            case 2:
                reduceKernel<2><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_temp);
                break;
            case 1:
                reduceKernel<1><<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_temp);
                break;
        }
        
        temp = (double) N / (dimBlock.x * 2);
        N = ceil(temp);

        dimGrid.x = (dimGrid.x > dimBlock.x) || (dimGrid.x == 1) ? (dimGrid.x / dimBlock.x) : 1;

        g_input = g_temp;
    }
    swapKernel<<<1, 1>>>(size, g_list, g_temp);

    if(size > 2){
        sortBySelection(size-1, g_list, g_temp);
    }
    

    return;
}

/* Get the number of blocks and threads per block */
void getGridComposition(int size, unsigned int* blocks, unsigned int* threads){

    *threads = THREADS;
    *blocks = 1;

    while (((*blocks) * (*threads) * 2) < size){
        *blocks<<=1;
    }

    if(*blocks == 1){
        while( ( *threads * 2 ) < size){
            *threads <<=1;
        }
    }

    return;
}

void printResults(DATATYPE *sorted_list)
{
    for (int i = 0; i < SIZE; i++)
    {
        printf("%d\n", sorted_list[i].value);
    }
    printf("\n");
    return;
}

int checkResults(DATATYPE *sorted_list){
    unsigned int check = 1;
    unsigned int index = SIZE;
    unsigned int i;
    for (i = 1; i < (SIZE + 1); i++)
    {
        if(sorted_list[i-1].value != i)
            check = 0;
        if(index == SIZE)
            index = i - 1;
    }

    if(check)
        printf("Resultados correctos!\n");
    else
        printf("Resultados incorrectos!\n");

    printf("Ultimo Indice: %d\n", i);
    printf("%d en %d", sorted_list[index].value, index);
    return check;
    
}

int main(void)
{
    DATATYPE *list, *list_g, *list_g_o;

    srand(time(NULL));

    list = (DATATYPE *)malloc(SIZE * sizeof(DATATYPE));
    if(list == NULL){
        printf("Error alocando memoria.\n");
        exit(0);
    }

    cudaMalloc((void**)&list_g, SIZE * sizeof(DATATYPE));
    cudaMalloc((void**)&list_g_o, SIZE * sizeof(DATATYPE));

    for (int i = 0; i < SIZE; i++)
    {
        //list[i] = rand() % 100;
        list[i].value = SIZE - i;
        list[i].index = i;
    }

    cudaMemcpy(list_g, list, SIZE * sizeof(DATATYPE), cudaMemcpyHostToDevice);

    // Perform Reduce Operation
    sortBySelection(SIZE, list_g, list_g_o);

    cudaMemcpy(list,list_g , SIZE * sizeof(DATATYPE), cudaMemcpyDeviceToHost);

    if (PRINT)
    {
        printResults(list);
    }
    if(CHECK)
    {
        checkResults(list);
    }

    cudaFree(list_g);
    cudaFree(list_g_o);
    free(list);
}