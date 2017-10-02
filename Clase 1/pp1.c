#include <omp.h>
#include "stdio.h"
#define SIZE 40
#define THREADS 4

int main(){
	int i,j, hilo, sumatotal;
	int a[SIZE], b[SIZE], c[SIZE], suma[THREADS];

	printf("\n");

	for(i=0; i<SIZE; i++){
		a[i] = 1;
		b[i] =1;
		c[i] = 0;
	}

	for (i=0; i<THREADS; i++)
	{
		suma[i]=0;
	}

	omp_set_num_threads(THREADS);

#pragma omp parallel private(hilo) shared(j, a, b, c, suma)
	{
		j = omp_get_thread_num();
		hilo = omp_get_thread_num();
#pragma omp barrier
		printf("i: %d - j: %d\n",hilo, j);
#pragma omp barrier

#pragma omp for //lastprivate(suma)
		for (i = 0; i < SIZE; i++){
			c[i] = a[i] + b[i];
			//printf("Escribi en posicion %d por hilo %d\n", i, hilo);
			suma[hilo] = suma[hilo] + c[i];
		}

		//printf("La suma del hilo %d es %d\n", hilo, suma);
	}
	
	sumatotal = 0;
	for(i = 0; i<THREADS; i++){
		sumatotal = sumatotal + suma [i];
	}

	printf ("La suma total es %d\n", sumatotal);
}