#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


#define TAMANO 2000
#define CACHE_BLOCK_SIZE 64
int main(){

	double start_time = omp_get_wtime();
	int i, j, k;

	/* i = filas - j = columnas */

	/* Alojo Memoria */
	/*
	Ejemplo:
	int** mat = (int**)malloc(rows * sizeof(int*))

	for (int index=0;index<row;++index)
	{
		mat[index] = (int*)malloc(col * sizeof(int));
	}
	*/
	float ** matrizA;
	float ** matrizB;
	float ** matrizC;
	matrizA = malloc(TAMANO*sizeof(float*));
	matrizB = calloc(TAMANO, sizeof(float*));
	matrizC = calloc(TAMANO, sizeof(float*));
	for(i = 0; i < TAMANO ; i++){
		matrizA[i] = malloc(TAMANO*sizeof(float));
		matrizB[i] = calloc(TAMANO, sizeof(float));
		matrizC[i] = calloc(TAMANO, sizeof(float));
	}

	/* Inicializo Matrices */
	/* Row-major order */
	for(i=0; i<TAMANO; i++)
		matrizB[i][i] = 1;
	for(i = 0; i<TAMANO; i++){
		for(j = 0; j<TAMANO; j++){
			matrizA[i][j] =1;
			matrizB[i][j] =1;
		}
	}

	/* Hago la Multiplicacion */
	/* Hay que hacer seis bucles */
	
	/*
	for(i = 0; i<TAMANO; i++){
		for(j = 0; j<TAMANO; j++){
			for(k = 0; k<TAMANO; k++){
				matrizC[i][j] += matrizA[i][k] * matrizB[k][j];
			}
		}
	}
	*/

	#pragma omp simd
	for(i = 0; i<TAMANO; i++){
		for(k = 0; k<TAMANO; k++){
			for(j = 0; j<TAMANO; j++){
				matrizC[i][j] += matrizA[i][k] * matrizB[k][j];
			}
		}
	}
	


	/* Mostrar Resultados */
	/*
	for(i = 0; i<TAMANO; i++){
		printf("\n");
		for(j = 0; j<TAMANO; j++){
			printf("%f    ", matrizC[i][j]);
		}
	}
	*/


	/* Libero Memoria */
	for(i = 0; i < TAMANO ; i++){
		free(matrizA[i]);
		free(matrizB[i]);
		free(matrizC[i]);
	}
	free(matrizA);
	free(matrizB);
	free(matrizC);
	

	double time = omp_get_wtime() - start_time;
	printf("El tiempo de ejecucion es: %lf segundos\n", time);

	return 1;
}
