#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 40


void sortArray(int * unsorted, int size)
{
    int max;
    int temp;
    int i;
    max = 0;
    for( i = 1; i < size; i++)
    {
        if(unsorted[max] < unsorted[i])
        {
            max = i;
        }
    }
    temp = unsorted[max];
    unsorted[max] = unsorted[size-1];
    unsorted[size-1] = temp;

    if (size > 1)
        sortArray(unsorted, size-1);
    
    return;
}

int main(int argc, char* argv[]){
    int unsorted[SIZE];
    int sorted[SIZE];
    int r, i;

    srand(time(NULL));

    for(i = 0; i<SIZE; i++)
    {
        unsorted[i] = rand() % 10;
    }

    sortArray(unsorted, SIZE);

    for(i = 0; i<SIZE; i++)
    {
        printf("%d ", unsorted[i]);
    }

}
