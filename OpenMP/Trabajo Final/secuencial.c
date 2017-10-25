#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


#define TAMANO 500
#define CACHE_BLOCK_SIZE 64
#define TILE_SIZE 256
int main(){

	double start_time = omp_get_wtime();
	int i, j, k, ia, ja, ka, imax, jmax, kmax;

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
	/*
	for(i=0; i<TAMANO; i++)
		matrizB[i][i] = 1;
	*/
	for(i = 0; i<TAMANO; i++){
		for(j = 0; j<TAMANO; j++){
			matrizA[i][j] =1;
			matrizB[i][j] =1;
		}
	}

	/* Hago la Multiplicacion */
	/* Hay que hacer seis bucles */
	
	/*
	#pragma omp simd
	for(i = 0; i<TAMANO; i++){
		for(k = 0; k<TAMANO; k++){
			for(j = 0; j<TAMANO; j++){
				matrizC[i][j] += matrizA[i][k] * matrizB[k][j];
			}
		}
	}
	*/
	
	#pragma omp simd
	for(i = 0; i<TAMANO; i+= TILE_SIZE){
		imax = i + TILE_SIZE > TAMANO ? TAMANO : i + TILE_SIZE;
		for(k = 0; k<TAMANO; k+= TILE_SIZE){
			kmax = k + TILE_SIZE > TAMANO ? TAMANO : k + TILE_SIZE;
			for(j = 0; j<TAMANO; j+= TILE_SIZE){
				jmax = j + TILE_SIZE > TAMANO ? TAMANO : j + TILE_SIZE;
				for(ia=i; ia<imax; ia++){
					for(ka=k; ka<kmax; ka++){
						for(ja=j; ja<jmax; ja++){
							matrizC[ia][ja] += matrizA[ia][ka] * matrizB[ka][ja];
						}
					}
				}
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
