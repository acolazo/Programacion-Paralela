#include <stdio.h>
#include <time.h>
#include <math.h>

#define SIZE 26
#define THREADS 4 //best value = 256
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

/* Kernel reduction at block level */
__global__ void reduceKernel(int size, DATATYPE *g_input, DATATYPE *g_output)
{
    extern __shared__ DATATYPE sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int gid = (blockIdx.x*blockDim.x) + tid;
    if(gid < size)
        sdata[tid] = g_input[gid];  
    __syncthreads();

    //do reduction in shared mem
    
    for(unsigned int s=blockDim.x/2; s > 0; s/=2){
        if((gid + s)<size && tid < s){
            sdata[tid] = sdata[tid].value > sdata[tid + s].value ? sdata[tid] : sdata[tid + s];
        }
        __syncthreads();
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
    //printf("Iteracion %d\n", SIZE - size + 1);
    while(dimGrid.x > 0){
        reduceKernel<<<dimGrid, dimBlock, N * sizeof(DATATYPE)>>>(N, g_input, g_temp);

       
        //printf("Grid: %d, N: %d\n", dimGrid.x, N);
        

        //N = dimGrid.x;
        temp = (double) N / dimBlock.x;
        N = ceil(temp);

        //dimGrid.x>>=1;
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

    while (((*blocks) * (*threads) ) < size){
        *blocks<<=1;
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

/* Check if results are correct */
int checkResults(DATATYPE *sorted_list){
    unsigned int check = 1;
    for (unsigned int i = 1; i < (SIZE + 1); i++)
    {
        if(sorted_list[i-1].value != i)
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