#include <stdio.h>

int main(int argc, char **argv)
{
    int x = 10;
    float y = 3.1415926;

    printf("Hello world\n");
    printf("Tall: %d %.2f, args: %d\n", x, y, argc - 1);
}