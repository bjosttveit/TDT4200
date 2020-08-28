#include <stdio.h>
#include <stdlib.h>

int main()
{
    int x = 10;
    int *array = NULL;
    array = malloc(sizeof(int) * x);

    for (int i = 0; i < x; i++) {
        array[i] = i;
        printf("%d\n", i);
    }
    free(array);
    array = NULL;
}