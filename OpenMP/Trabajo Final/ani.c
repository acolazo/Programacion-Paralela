#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


int main(int argc, char* argv[]){

	int TAMANO;
	int TILE_SIZE; //256

	double start_time = omp_get_wtime();
	int i, j, k, ia, ja, ka, imax, jmax, kmax;


	if( argc == 3 ) {
		sscanf (argv[1],"%d",&TAMANO);	
		sscanf (argv[2],"%d",&TILE_SIZE);	
	}
	else if( argc > 3 ) {
		exit(0);
	}
	else {
		printf("One argument expected.\n");
	}
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
	if (matrizA==NULL || matrizB==NULL || matrizC==NULL){
		printf("ERROR ALOCANDO MEMORIA\n");
		exit(0);

	}

	for(i = 0; i < TAMANO ; i++){
		matrizA[i] = malloc(TAMANO*sizeof(float));
		matrizB[i] = calloc(TAMANO, sizeof(float));
		matrizC[i] = calloc(TAMANO, sizeof(float));

		if (matrizA[i]==NULL || matrizB[i]==NULL || matrizC[i]==NULL){
			printf("ERROR ALOCANDO MEMORIA\n");
			exit(0);
		}
		
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
	printf("%lf,", time);

	return 1;
}
