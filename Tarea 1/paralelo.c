#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
//#include <time.h>

#define SIZE 40
#define THREADS 4
#define TEST 1000

int main(){
	

// code

	/*
	clock_t start;
	clock_t end;
	double elapsed_time;
	srand(time(NULL));   
	*/
	int tamanio, i, j, hilo;

	printf("Por ejemplo: Para el tamanio 1000, ingresar 1\n");
	while(tamanio != 1000 && tamanio != 2000 && tamanio != 4000 & tamanio != 8000 & tamanio != TEST){
		printf("Elija el tamanio de matriz (1000, 2000, 4000 o 8000)\n");

		scanf("%d", &tamanio);
	}

	//start=clock();
	int * matriz;
	int * filas;
	int * columnas;

	matriz = (int*)calloc(tamanio*tamanio, sizeof(int));
	filas = (int*)calloc(tamanio, sizeof(int));
	columnas = (int*)calloc(tamanio, sizeof(int));


	printf("Suma de filas	Suma de Columnas\n");
	double start_time = omp_get_wtime();

	omp_set_num_threads(THREADS);
#pragma omp parallel private(hilo, j) shared(matriz, filas, columnas)
	{

#pragma omp for
		for(i=0; i<tamanio; i++){
		//filas[i] = 0;
		//columnas[i] = 0;
			for(j=0; j<tamanio; j++){
			//matriz[i][j] = rand();
				matriz[i+tamanio*j] = 2;
			}
		}

#pragma omp for
		for(i=0; i<tamanio; i++){
			for(j=0; j<tamanio; j++){
				filas[i] = filas[i] + matriz[i+tamanio*j];
				columnas[i] = columnas[i] + matriz[tamanio*j+i];
			}
		}

} //Termina la parte paralela

double time = omp_get_wtime() - start_time;
//end = clock();
//elapsed_time = (end-start)/(double)CLOCKS_PER_SEC ;

for(i=0; i<tamanio; i++){
	printf("Fila %d: %d 	Columna %d: %d\n", i, filas[i], i, columnas[i]);
}

printf("\nEl tiempo que paso es de: %lf\n", time);

free(matriz);
free(filas);
free(columnas);

//printf("\nEl tiempo que paso es de: %lf\n", elapsed_time);

return;
}