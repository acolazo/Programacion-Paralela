/* = &a[0][0];
ptr = &a[0][0];
r = &a[0][0];
= &a[0][0];
= &a[0][0]; /
    &[0]([0] + i * COLS + j) ble
tos de cada una de las matrices en 4 bloques cuadrados, cada uno de los cuales será poseido por cada uno de los procesos.

Por ejemplo, para matrices A, B y C de 2000x2000, donde C= A x B , cada uno de los procesos es dueño de un bloque de A 
de 500 x 500, de uno de B y de uno de C de igual tamaño. El proceso dueño de cada bloque de C es quien realiza el cómputo, 
y como C está distribuida entre los 4 procesos, todos computan en paralelo recibiendo los bloques desde los otros procesos.

Haga un algoritmo que utilice primitivas bloqueantes, y otro no bloqueantes y verifique rendimiento. 

Plus: puede utilizar openmp en la parte del procesamiento, pero debe ser performante.
*/

#include "mpi.h"
#include "omp.h"
#include <stdio.h>
#include <stdlib.h>

#define RANGE 4
#define PROCS 4
#define COLS RANGE / 2
#define ROWS RANGE / 2

struct metaSend
{
  int taskid;
  int sendto[2];
  int receivefrom[2];
  int sender; //first to send
};

int main(int argc, char *argv[])
{
  int numtasks,          /* number of tasks in partition */
      taskid,            /* a task identifier */
      source,            /* task id of message source */
      dest,              /* task id of message destination */
      g, i, j, k, h, rc, /* misc */
      n, alg;            /* matrix size y frac = n /p */

  double a[ROWS][COLS], /* matrix A to be multiplied */
      b[ROWS][COLS],    /* matrix B to be multiplied */
      bw[ROWS][COLS],   /* buffer to receive data */
      c[ROWS][COLS];    /* result matrix C */

  double inib, finb, inic, finc, inig, fing;

  MPI_Status status;
  MPI_Request reqs[PROCS];
  MPI_Status statuss[PROCS];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  if (numtasks != PROCS)
  {
    printf("Solo se necesitan %d, y hay %d Saliendo...\n", PROCS, numtasks);
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
  }

  //Inicializar Matrices
  for (i = 0; i < ROWS; i++)
  {
    for (j = 0; j < COLS; j++)
    {
      /*
      a[i][j] = 1;
      b[i][j] = 1;
      c[i][j] = 0;
      */

      a[i][j] = 5;
      c[i][j] = 0;
      if (i == j && (taskid == 0 || taskid == 1))
      {
        b[i][j] = 1;
      }
      else
      {
        b[i][j] = 0;
      }
    }
  }

  //Cada proceso multiplica su parte
  for (i = 0; i < ROWS; i++)
    for (k = 0; k < ROWS; k++) //ROWS == COLS
      for (j = 0; j < COLS; j++)
        c[i][j] += a[i][k] * b[k][j];

  

  /* */
  if (0)
  {
    printf("Proceso %d\n", taskid);
    for (i = 0; i < ROWS; i++)
    {
      for (j = 0; j < COLS; j++)
      {

        printf("%f ", c[i][j]);
      }
      printf("\n");
    }
  }
  /* */
  //Creo metadatos sobre a quien enviar y de quien recibir la informacion
  int sendi, recvi;
  int sender[8] = {0, 3, 0, 3, 1, 2, 1, 2};
  int receiver[8] = {1, 2, 2, 1, 0, 3, 3, 0};
  struct metaSend metadata; //metadata for sending and receiving information
  sendi = 0;
  recvi = 0;
  metadata.taskid = taskid;
  for (i = 0; i < 8; i++)
  {
    if (sender[i] == taskid)
    {
      metadata.sendto[sendi] = receiver[i];
      sendi++;
    }

    if (receiver[i] == taskid)
    {
      metadata.receivefrom[recvi] = sender[i];
      recvi++;
    }
  }
  if (taskid == sender[0] || taskid == sender[1])
    metadata.sender = 1;
  else
    metadata.sender = 0;


  printf("Metada Proceso %d: Send: %d %d Receive: %d %d \n", taskid, metadata.sendto[0], metadata.sendto[1], metadata.receivefrom[0], metadata.receivefrom[1]);
  //Ahora cada proceso debe enviarle a los otros los bloques que les faltan. Las llamadas son bloqueantes
  //En la primera vuelta ptr apunta a la matriz A, y en la segunda a la matriz B

  double *ptr;
  ptr = &a[0][0];
  for (k = 0; k < 2; k++)
  {
    if (metadata.sender)
    {
      MPI_Send(ptr, ROWS * COLS, MPI_DOUBLE, metadata.sendto[k], 1, MPI_COMM_WORLD);
      MPI_Recv(ptr, ROWS * COLS, MPI_DOUBLE, metadata.receivefrom[k], 1, MPI_COMM_WORLD, &status);
    }
    else
    {
      MPI_Recv(&bw, ROWS * COLS, MPI_DOUBLE, metadata.receivefrom[k], 1, MPI_COMM_WORLD, &status);
      MPI_Send(ptr, ROWS * COLS, MPI_DOUBLE, metadata.sendto[k], 1, MPI_COMM_WORLD);
      for (i = 0; i < ROWS; i++)
        for (j = 0; j < COLS; j++)
          *(ptr + i * COLS + j) = bw[i][j];
    }
    ptr = &b[0][0];
  }

  //Cada proceso continua multiplicando
  for (i = 0; i < ROWS; i++)
    for (k = 0; k < ROWS; k++) //ROWS == COLS
      for (j = 0; j < COLS; j++)
        c[i][j] += a[i][k] * b[k][j];

  //Checkear que las matrices esten bien.
  int err = 0;
  if (1)
  {

    printf("Proceso %d\n", taskid);
    for (i = 0; i < ROWS; i++)
    {
      for (j = 0; j < COLS; j++)
      {
        if (c[i][j] != 5)
        {
          err = 1;
        }
        printf("%f ", c[i][j]);
      }
      printf("\n");
    }

    if (err)
      printf("Hubo un error en proceso %d\n", taskid);
    else
      printf("Todo estuvo bien en proceso %d\n", taskid);
  }
  MPI_Finalize();
}
