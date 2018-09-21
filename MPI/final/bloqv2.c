#include "mpi.h"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#define MEASURE_PHASES 1
#define CHECK 1
#define RANGE 8000
#define NUM_THREADS 4
#define PROCS 4
#define COLS RANGE / 2
#define ROWS RANGE / 2
#define TAG 10
#define TIMES_TO_INTERRUPT_MULTIPLICATION 4
#define ROUNDS ROWS / TIMES_TO_INTERRUPT_MULTIPLICATION
#define OPTION 2
/* Opcion 1: Matriz A y B estan rellenas de 1's */
/* Opcion 2: Matriz A esta rellena de 5's. Matriz B es la matriz identidad. */
#define MATRIX_VALUE_OPTION_2 5
#define PRINT_RESULTS 0

struct metadata
{
    int taskid;
    int sendto[2];
    int receivefrom[2];
    int sender;
};

void allocateMemory(double *ptr)
{
    ptr = (double *)calloc(ROWS * COLS, sizeof(double));
    if (ptr == NULL)
    {
        printf("ERROR ALOCANDO MEMORIA\n");
        exit(0);
    }
}
void freeMemory(double *ptr)
{
    free(ptr);
    ptr = NULL;
}

void createMetaData(struct metadata *metadata, int taskid)
{
    /* sentto[0]: destination for matrix A */
    /* sentto[1]: destination for matrix B */
    /* receivefrom[0]: source for matrix A */
    /* receivefrom[0]: source for matrix B */
    int sendi, recvi;
    int sender[8] = {0, 3, 0, 3, 1, 2, 1, 2};
    int receiver[8] = {1, 2, 2, 1, 0, 3, 3, 0};
    int i;
    sendi = 0;
    recvi = 0;
    metadata->taskid = taskid;
    for (i = 0; i < 8; i++)
    {
        if (sender[i] == taskid)
        {
            metadata->sendto[sendi] = receiver[i];
            sendi++;
        }

        if (receiver[i] == taskid)
        {
            metadata->receivefrom[recvi] = sender[i];
            recvi++;
        }
    }
    if (taskid == sender[0] || taskid == sender[1])
        metadata->sender = 1;
    else
        metadata->sender = 0;
}

int multiplicarMatrices(double *a, double *b, double *c)
{
    int i, j, k;
#pragma omp for
    for (i = 0; i < ROWS; i++)
    {
        for (k = 0; k < ROWS; k++)
        {
            for (j = 0; j < ROWS; j++)
            {
                c[i * COLS + j] += a[i * COLS + k] * b[k * COLS + j];
            }
        }
    }
    return 1;
}

int multiplicarMatricesPreemptive(double *a, double *b, double *c)
{
    static int i = 0;
    int j, k;
    int rounds;
    rounds = 0;
    while (i < ROWS)
    {
        if (rounds == ROUNDS)
        {
            return 0;
        }
        else
        {
            rounds++;
        }

        for (k = 0; k < ROWS; k++)
        {
            for (j = 0; j < ROWS; j++)
            {
                c[i * COLS + j] += a[i * COLS + k] * b[k * COLS + j];
            }
        }
        i++;
    }
    i = 0;
    j = 0;
    k = 0;
    return 1;
}

int copyData(double *copyfrom, double *copyto)
{
    double *temp;

    temp = copyfrom;
    copyfrom = copyto;
    copyto = copyfrom;
    return 1;
    /*
    int i, j;
//pragma omp for
    for (i = 0; i < ROWS; i++)
        for (j = 0; j < COLS; j++)
            copyto[i * COLS + j] = copyfrom[i * COLS + j];

    return 1;
    */
}

int checkResults(double *results, int option)
{
    int i, j, correct_data;
    correct_data = 1;
    switch (option)
    {
    case 1:
    {
        for (i = 0; i < ROWS; i++)
            for (j = 0; j < COLS; j++)
                if (results[i * COLS + j] != RANGE)
                    correct_data = 0;
        break;
    }
    case 2:
    {
        for (i = 0; i < ROWS; i++)
            for (j = 0; j < COLS; j++)
                if (results[i * COLS + j] != MATRIX_VALUE_OPTION_2)
                    correct_data = 0;
        break;
    }
    }
    return correct_data;
}

int readData(double *a, double *b, double *c, int taskid, int option)
{
    /* Opcion 1: Matriz A y B estan rellenas de 1's */
    /* Opcion 2: Matriz A esta rellena de 5's. Matriz B es la matriz identidad. */
    /* No esta claro si el proceso principal envia a cada parte los datos, o cada uno comienza con sus datos */
    int i, j;
    for (i = 0; i < ROWS; i++)
    {
        for (j = 0; j < COLS; j++)
        {
            switch (option)
            {

            case 1:
            {
                a[i * COLS + j] = 1;
                b[i * COLS + j] = 1;
                c[i * COLS + j] = 0;
                break;
            }

            case 2:
            {
                a[i * COLS + j] = MATRIX_VALUE_OPTION_2;
                c[i * COLS + j] = 0;
                if (i == j && (taskid == 0 || taskid == 1))
                {
                    b[i * COLS + j] = 1;
                }
                else
                {
                    b[i * COLS + j] = 0;
                }
                break;
            }

            default:
            {
                printf("ERROR\n");
                break;
            }
            }
        }
    }
    return 1;
}

int sendData(double *ptr, int destination, MPI_Request *req_send, MPI_Status *statuss, int taskid, int *pending)
{
    /* Opcion 1: Enviar matriz A */
    /* Opcion 2: Enviar matriz B */

    //
    MPI_Ssend(ptr, ROWS * COLS, MPI_DOUBLE, destination, TAG, MPI_COMM_WORLD);
    return 1;
}

int receiveData(double *ptr, int source, MPI_Request *req_recv, MPI_Status *statuss, int taskid)
{
    /*
    Receive_data
    return test_data;
  */
    MPI_Recv(ptr, ROWS * COLS, MPI_DOUBLE, source, TAG, MPI_COMM_WORLD, statuss + taskid);
    return 1;
}

int main(int argc, char *argv[])
{
    int numtasks,    /* number of tasks in partition */
        taskid,      /* a task identifier */
        source,      /* task id of message source */
        dest,        /* task id of message destination */
        i, j, k, rc; /* misc */
    struct metadata metadata;

    double *a, *b, *c, *bw;
    int comm_status;

    MPI_Request req_send_a[PROCS];
    MPI_Request req_send_b[PROCS];
    MPI_Request req_recv[PROCS];
    MPI_Status statuss[PROCS];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    /* Check MPI Process number to be correct, else aborts */

    if (numtasks != PROCS)
    {
        printf("Solo se necesitan %d, y hay %d Saliendo...\n", PROCS, numtasks);
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    /* Allocate memory */

    a = (double *)calloc(ROWS * COLS, sizeof(double));
    if (a == NULL)
    {
        printf("ERROR ALOCANDO MEMORIA\n");
        exit(0);
    }
    b = (double *)calloc(ROWS * COLS, sizeof(double));
    if (b == NULL)
    {
        printf("ERROR ALOCANDO MEMORIA\n");
        exit(0);
    }
    c = (double *)calloc(ROWS * COLS, sizeof(double));
    if (c == NULL)
    {
        printf("ERROR ALOCANDO MEMORIA\n");
        exit(0);
    }
    bw = (double *)calloc(ROWS * COLS, sizeof(double));
    if (bw == NULL)
    {
        printf("ERROR ALOCANDO MEMORIA\n");
        exit(0);
    }
    /* Each process gets it's data */
    readData(a, b, c, taskid, OPTION);

    /* Each process gets metadata with sending and receiving information */
    createMetaData(&metadata, taskid);

    /* This variables indicate states */
    int sendA, sendB, calculateC1, calculateC2, copyA, copyB, recvA, recvB;

    /* SendA and sendB pending variables */
    int pendingA = 0;
    int pendingB = 0;

    double start, end;
    double start1, end1, start2, end2, start3, end3, start2a, end2a;

    start = MPI_Wtime();
    start1 = MPI_Wtime();
    /* Loop */

#pragma omp parallel shared(a, b, c, bw, taskid, statuss, req_send_a, req_send_b, req_recv) num_threads(NUM_THREADS)
    {
        calculateC1 = multiplicarMatrices(a, b, c);

#pragma omp single
        {
            end1 = MPI_Wtime();
            start2 = MPI_Wtime();
            if (metadata.sender)
            {
                MPI_Send(a, ROWS * COLS, MPI_DOUBLE, metadata.sendto[0], TAG, MPI_COMM_WORLD);
                MPI_Recv(bw, ROWS * COLS, MPI_DOUBLE, metadata.receivefrom[0], TAG, MPI_COMM_WORLD, statuss + taskid);
            }
            else
            {
                MPI_Recv(bw, ROWS * COLS, MPI_DOUBLE, metadata.receivefrom[0], TAG, MPI_COMM_WORLD, statuss + taskid);
                MPI_Send(a, ROWS * COLS, MPI_DOUBLE, metadata.sendto[0], TAG, MPI_COMM_WORLD);
            }
            
            //pragma omp barrier
            copyA = copyData(bw, a);
            end2 = MPI_Wtime();
            start2a = MPI_Wtime();
        }
#pragma omp barrier

#pragma omp single
        {
            if (metadata.sender)
            {
                MPI_Send(b, ROWS * COLS, MPI_DOUBLE, metadata.sendto[1], TAG, MPI_COMM_WORLD);
                MPI_Recv(bw, ROWS * COLS, MPI_DOUBLE, metadata.receivefrom[1], TAG, MPI_COMM_WORLD, statuss + taskid);
            }
            else
            {
                MPI_Recv(bw, ROWS * COLS, MPI_DOUBLE, metadata.receivefrom[1], TAG, MPI_COMM_WORLD, statuss + taskid);
                MPI_Send(b, ROWS * COLS, MPI_DOUBLE, metadata.sendto[1], TAG, MPI_COMM_WORLD);
            }
            //end2 = MPI_Wtime();
            end2a = MPI_Wtime();
            start3 = MPI_Wtime();
        }

#pragma omp barrier
        calculateC2 = multiplicarMatrices(a, bw, c);
    }
    end3 = MPI_Wtime();
    end = MPI_Wtime();
    /* Check Results */
    if (CHECK && checkResults(c, OPTION))
    {
        printf("Results in process %d were correct and it took %f seconds\n", taskid, end - start);
    }
    else if (CHECK)
    {
        printf("Results in process %d were incorrect\n", taskid);
    }

    if (MEASURE_PHASES)
    {
        printf("Process %d took %f seconds in phase 1\n", taskid, end1 - start1);
        printf("Process %d took %f seconds in phase 2\n", taskid, end2 - start2);
        printf("Process %d took %f seconds in phase 2a\n", taskid, end2a - start2a);
        printf("Process %d took %f seconds in phase 3\n", taskid, end3 - start3);
    }

    /* Print Results */
    int err = 0;
    if (PRINT_RESULTS)
    {

        printf("Proceso %d\n", taskid);
        for (i = 0; i < ROWS; i++)
        {
            for (j = 0; j < COLS; j++)
            {
                if (c[i * COLS + j] != 5)
                {
                    err = 1;
                }
                printf("%f ", c[i * COLS + j]);
            }
            printf("\n");
        }
    }

    /* Freed memory */
    freeMemory(a);
    freeMemory(b);
    freeMemory(c);
    freeMemory(bw);

    MPI_Finalize();
}
