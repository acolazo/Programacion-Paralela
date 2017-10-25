#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


#define THREADS 2
#define TAMANO 500
#define TILE_SIZE 256
int main(){

	double start_time = omp_get_wtime();
	int i, j, k;
	int ia, ja, ka;
	int imax, jmax, kmax;

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
	matrizB = malloc(TAMANO* sizeof(float*));
	matrizC = malloc(TAMANO* sizeof(float*));

	if (matrizA==NULL || matrizB==NULL || matrizC==NULL){
		printf("ERROR ALOCANDO MEMORIA\n");
		exit(0);
	}

	//omp_set_num_threads(THREADS);
	#pragma omp parallel shared(matrizA, matrizB, matrizC) private(i, j, k, ia, ja, ka, imax, kmax, jmax)
	{

	#pragma omp for
	for(i = 0; i < TAMANO ; i++){
		matrizA[i] = malloc(TAMANO*sizeof(float));
		matrizB[i] = malloc(TAMANO*sizeof(float));
		matrizC[i] = calloc(TAMANO, sizeof(float));

		if (matrizA[i]==NULL || matrizB[i]==NULL || matrizC[i]==NULL){
			printf("ERROR ALOCANDO MEMORIA\n");
			exit(0);
		}
	}

	/* Inicializo Matrices */
	/* Row-major order */
	/*
	#pragma omp for
	#pragma omp simd
	for(i=0; i<TAMANO; i++)
		matrizB[i][i] = 1;
	*/

	#pragma omp for
	for(i = 0; i<TAMANO; i++){
		for(j = 0; j<TAMANO; j++){
			matrizA[i][j] =1;
			matrizB[i][j] =1;
		}
	}

	/* Hago la Multiplicacion */
	/* Hay que hacer seis bucles */
	/*
	for(i = 0; i<TAMANO; i+= TILE_SIZE){
		for(j = 0; j<TAMANO; j+= TILE_SIZE){
			for(k = 0; k<TAMANO; k+= TILE_SIZE){
				for(ia=i; ia<i+TILE_SIZE; ia++){
					for(ja=j; ja<j+TILE_SIZE; ja++){
						for(ka=k; ka<k+TILE_SIZE; ka++){
							matrizC[ia][ja] += matrizA[ia][ka] * matrizB[ka][ja];
						}
					}
				}
			}
		}
	}
	*/
	//simd aligned (matrizC, matrizA, matrizB: 32)

	/*
	if(i + TILE_SIZE > TAMANO)
			imax=TAMANO;
		else
			imax=i+TILE_SIZE;
	*/
	#pragma omp for
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
	#pragma omp for
	for(i = 0; i < TAMANO ; i++){
		free(matrizA[i]);
		free(matrizB[i]);
		free(matrizC[i]);
	}

}
	free(matrizA);
	free(matrizB);
	free(matrizC);
	

	double time = omp_get_wtime() - start_time;
	//printf("El tiempo de ejecucion es: %lf segundos\n", time);
	printf("%lf,", time);
	/*
	int aux = THREADS;
	printf("Number THREADS: %d\n", aux);
	*/
	

	return 1;
}
