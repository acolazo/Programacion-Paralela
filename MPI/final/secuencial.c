#include <omp.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#define RANGE 8000
#define NUM_THREADS 4
#define PROCS 4
#define COLS RANGE
#define ROWS RANGE
#define TAG 10
#define TIMES_TO_INTERRUPT_MULTIPLICATION 4
#define ROUNDS ROWS / TIMES_TO_INTERRUPT_MULTIPLICATION
#define OPTION 2
#define MATRIX_VALUE_OPTION_2 5
#define PRINT_RESULTS 0

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

int multiplicarMatrices(double *a, double *b, double *c)
{
  int i, j, k;

  #pragma omp for
  for (i = 0; i < RANGE; i++)
  {
    for (k = 0; k < RANGE; k++)
    {
      for (j = 0; j < RANGE; j++)
      {
        c[i * RANGE + j] += a[i * RANGE + k] * b[k * RANGE + j];
      }
    }
  }
  return 1;
}


int checkResults(double *results, int option)
{
  int i, j, correct_data;
  correct_data = 1;
  switch (option)
  {
  case 1:
  {
    for (i = 0; i < RANGE; i++)
      for (j = 0; j < RANGE; j++)
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

int readData(double *a, double *b, double *c, int option)
{
  /* Opcion 1: Matriz A y B estan rellenas de 1's */
  /* Opcion 2: Matriz A esta rellena de 5's. Matriz B es la matriz identidad. */
  /* No esta claro si el proceso principal envia a cada parte los datos, o cada uno comienza con sus datos */
  int i, j;
  for (i = 0; i < RANGE; i++)
  {
    for (j = 0; j < RANGE; j++)
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
        a[i * RANGE + j] = MATRIX_VALUE_OPTION_2;
        c[i * RANGE + j] = 0;
        if (i == j)
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




int main(int argc, char *argv[])
{
 
  double *a, *b, *c;


  /* Allocate memory */

  a = (double *)calloc(RANGE * RANGE, sizeof(double));
  if (a == NULL)
  {
    printf("ERROR ALOCANDO MEMORIA\n");
    exit(0);
  }
  b = (double *)calloc(RANGE * RANGE, sizeof(double));
  if (b == NULL)
  {
    printf("ERROR ALOCANDO MEMORIA\n");
    exit(0);
  }
  c = (double *)calloc(RANGE * RANGE, sizeof(double));
  if (c == NULL)
  {
    printf("ERROR ALOCANDO MEMORIA\n");
    exit(0);
  }
  
  /* Each process gets it's data */
  readData(a, b, c, OPTION);




  double start, end;
  start = omp_get_wtime();
  /* Loop */
#pragma omp parallel shared(a, b, c) num_threads(NUM_THREADS)
  {
  multiplicarMatrices(a, b, c);
  }
 
  end = omp_get_wtime();
  /* Check Results */
  if (checkResults(c, OPTION))
  {
    printf("Results in process 0 were correct and it took %f seconds\n", end - start);
  }
  else
  {
    printf("Results in process 0 were incorrect\n");
  }

  /* Print Results */
  int err = 0;
  int i, j;
  if (PRINT_RESULTS)
  {

    printf("Proceso 0\n");
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
}
