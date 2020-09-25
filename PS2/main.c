#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "libs/bitmap.h"

// Convolutional Kernel Examples, each with dimension 3,
// gaussian kernel with dimension 5

int sobelYKernel[] = {-1, -2, -1,
                      0, 0, 0,
                      1, 2, 1};

int sobelXKernel[] = {-1, -0, 1,
                      -2, 0, 2,
                      -1, 0, 1};

int laplacian1Kernel[] = {-1, -4, -1,
                          -4, 20, -4,
                          -1, -4, -1};

int laplacian2Kernel[] = {0, 1, 0,
                          1, -4, 1,
                          0, 1, 0};

int laplacian3Kernel[] = {-1, -1, -1,
                          -1, 8, -1,
                          -1, -1, -1};

int gaussianKernel[] = {1, 4, 6, 4, 1,
                        4, 16, 24, 16, 4,
                        6, 24, 36, 24, 6,
                        4, 16, 24, 16, 4,
                        1, 4, 6, 4, 1};

char *const kernelNames[] = {"SobelY", "SobelX", "Laplacian 1", "Laplacian 2", "Laplacian 3", "Gaussian"};
int *const kernels[] = {sobelYKernel, sobelXKernel, laplacian1Kernel, laplacian2Kernel, laplacian3Kernel, gaussianKernel};
unsigned int const kernelDims[] = {3, 3, 3, 3, 3, 5};
float const kernelFactors[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0 / 256.0};

int const maxKernelIndex = sizeof(kernelDims) / sizeof(unsigned int);

// Helper function to swap bmpImageChannel pointers

void swapImage(bmpImage **one, bmpImage **two)
{
  bmpImage *helper = *two;
  *two = *one;
  *one = helper;
}

// Apply convolutional kernel on image data
// Modified to use halos above and below image
void applyKernel(pixel **out, pixel **in, unsigned int width, unsigned int height, int *kernel, unsigned int kernelDim, float kernelFactor, pixel **topHalo, pixel **bottomHalo)
{
  unsigned int const kernelCenter = (kernelDim / 2);
  for (unsigned int y = 0; y < height; y++)
  {
    for (unsigned int x = 0; x < width; x++)
    {
      unsigned int ar = 0, ag = 0, ab = 0;
      for (unsigned int ky = 0; ky < kernelDim; ky++)
      {
        int nky = kernelDim - 1 - ky;
        for (unsigned int kx = 0; kx < kernelDim; kx++)
        {
          int nkx = kernelDim - 1 - kx;

          int yy = y + (ky - kernelCenter);
          int xx = x + (kx - kernelCenter);
          //If within image's width
          if (xx >= 0 && xx < (int)width)
          {
            //If within image's height. Use pixels from image as normal
            if (yy >= 0 && yy < (int)height)
            {
              ar += in[yy][xx].r * kernel[nky * kernelDim + nkx];
              ag += in[yy][xx].g * kernel[nky * kernelDim + nkx];
              ab += in[yy][xx].b * kernel[nky * kernelDim + nkx];
            }
            //If above image use pixels from topHalo
            else if (yy < 0)
            {
              ar += topHalo[kernelCenter + yy][xx].r * kernel[nky * kernelDim + nkx];
              ag += topHalo[kernelCenter + yy][xx].g * kernel[nky * kernelDim + nkx];
              ab += topHalo[kernelCenter + yy][xx].b * kernel[nky * kernelDim + nkx];
            }
            //If below image use pixels from bottomHalo
            else if (yy >= (int)height)
            {
              ar += bottomHalo[yy - height][xx].r * kernel[nky * kernelDim + nkx];
              ag += bottomHalo[yy - height][xx].g * kernel[nky * kernelDim + nkx];
              ab += bottomHalo[yy - height][xx].b * kernel[nky * kernelDim + nkx];
            }
          }
        }
      }
      if (ar || ag || ab)
      {
        ar *= kernelFactor;
        ag *= kernelFactor;
        ab *= kernelFactor;
        out[y][x].r = (ar > 255) ? 255 : ar;
        out[y][x].g = (ag > 255) ? 255 : ag;
        out[y][x].b = (ab > 255) ? 255 : ab;
      }
      else
      {
        out[y][x].r = 0;
        out[y][x].g = 0;
        out[y][x].b = 0;
      }
    }
  }
}

void help(char const *exec, char const opt, char const *optarg)
{
  FILE *out = stdout;
  if (opt != 0)
  {
    out = stderr;
    if (optarg)
    {
      fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
    }
    else
    {
      fprintf(out, "Invalid parameter - %c\n", opt);
    }
  }
  fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
  fprintf(out, "\n");
  fprintf(out, "Options:\n");
  fprintf(out, "  -k, --kernel     <kernel>        kernel index (0<=x<=%u) (2)\n", maxKernelIndex - 1);
  fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

  fprintf(out, "\n");
  fprintf(out, "Example: %s before.bmp after.bmp -i 10000\n", exec);
}

int main(int argc, char **argv)
{
  /*
    Parameter parsing, don't change this!
   */
  unsigned int iterations = 1;
  char *output = NULL;
  char *input = NULL;
  unsigned int kernelIndex = 2;
  int ret = 0;

  static struct option const long_options[] = {
      {"help", no_argument, 0, 'h'},
      {"kernel", required_argument, 0, 'k'},
      {"iterations", required_argument, 0, 'i'},
      {0, 0, 0, 0}};

  static char const *short_options = "hk:i:";
  {
    char *endptr;
    int c;
    int parse;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1)
    {
      switch (c)
      {
      case 'h':
        help(argv[0], 0, NULL);
        goto graceful_exit;
      case 'k':
        parse = strtol(optarg, &endptr, 10);
        if (endptr == optarg || parse < 0 || parse >= maxKernelIndex)
        {
          help(argv[0], c, optarg);
          goto error_exit;
        }
        kernelIndex = (unsigned int)parse;
        break;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg)
        {
          help(argv[0], c, optarg);
          goto error_exit;
        }
        break;
      default:
        abort();
      }
    }
  }

  if (argc <= (optind + 1))
  {
    help(argv[0], ' ', "Not enough arugments");
    goto error_exit;
  }

  unsigned int arglen = strlen(argv[optind]);
  input = calloc(arglen + 1, sizeof(char));
  strncpy(input, argv[optind], arglen);
  optind++;

  arglen = strlen(argv[optind]);
  output = calloc(arglen + 1, sizeof(char));
  strncpy(output, argv[optind], arglen);
  optind++;

  /*
    End of Parameter parsing!
   */

  int comm_sz;
  int my_rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  /*
    Create the BMP image and load it from disk.
   */
  double startTime;
  bmpImage *image = newBmpImage(0, 0);
  int width, height;

  if (my_rank == 0)
  {
    if (image == NULL)
    {
      fprintf(stderr, "Could not allocate new image!\n");
      goto error_exit;
    }

    if (loadBmpImage(image, input) != 0)
    {
      fprintf(stderr, "Could not load bmp image '%s'!\n", input);
      freeBmpImage(image);
      goto error_exit;
    }

    width = image->width;
    height = image->height;

    printf("Apply kernel '%s' on image with %u x %u pixels for %u iterations\n", kernelNames[kernelIndex], image->width, image->height, iterations);

    startTime = MPI_Wtime();
  }

  //Define MPI type for transfering pixels
  MPI_Datatype MPI_PIXEL;
  MPI_Type_contiguous(3, MPI_UNSIGNED_CHAR, &MPI_PIXEL);
  MPI_Type_commit(&MPI_PIXEL);

  //Only rank 0 has the image, but all ranks need dimensions to allocate buffers
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

  //Calculate height and start for each rank's row
  int *offsets = (int *)malloc(comm_sz * sizeof(int));
  int *counts = (int *)malloc(comm_sz * sizeof(int));

  int rowHeight = height / comm_sz;

  for (int i = 0; i < comm_sz; i++)
  {
    offsets[i] = i * rowHeight * width;
    counts[i] = rowHeight * width;
  }
  //Last rank takes the 'leftover' rows not divisble by the number of ranks
  counts[comm_sz - 1] = height * width - rowHeight * width * (comm_sz - 1);

  //Allocate buffers for my row
  bmpImage *rowImage = newBmpImage(width, counts[my_rank] / width);
  bmpImage *rowBuffer = newBmpImage(width, counts[my_rank] / width);

  //Distribute image in rows to all ranks
  MPI_Scatterv(image->rawdata, counts, offsets, MPI_PIXEL, rowImage->rawdata, counts[my_rank], MPI_PIXEL, 0, MPI_COMM_WORLD);

  //Get halowidth based on kernel size
  int const haloThickness = kernelDims[kernelIndex] / 2;
  //Check if rank is even or odd to prevent deadlock
  bool evenRank = (my_rank % 2 == 0);
  bool topRank = (my_rank == 0);
  bool bottomRank = (my_rank == comm_sz - 1);

  //Allocate buffers for sendRecv
  bmpImage *topSendHalo = newBmpImage(rowImage->width, haloThickness);
  bmpImage *bottomSendHalo = newBmpImage(rowImage->width, haloThickness);
  bmpImage *topRecvHalo = newBmpImage(rowImage->width, haloThickness);
  bmpImage *bottomRecvHalo = newBmpImage(rowImage->width, haloThickness);

  //Repeats every iteration
  for (unsigned int i = 0; i < iterations; i++)
  {
    //Copy data from my row to top halo to be sent out
    if (!topRank)
    {
      for (int y = 0; y < haloThickness; y++)
      {
        for (int x = 0; x < rowImage->width; x++)
        {
          topSendHalo->data[y][x].r = rowImage->data[y][x].r;
          topSendHalo->data[y][x].g = rowImage->data[y][x].g;
          topSendHalo->data[y][x].b = rowImage->data[y][x].b;
        }
      }
    }
    //Copy data from my row to bottom halo to be sent out
    if (!bottomRank)
    {
      for (int y = 0; y < haloThickness; y++)
      {
        for (int x = 0; x < rowImage->width; x++)
        {
          bottomSendHalo->data[y][x].r = rowImage->data[rowImage->height - haloThickness + y][x].r;
          bottomSendHalo->data[y][x].g = rowImage->data[rowImage->height - haloThickness + y][x].g;
          bottomSendHalo->data[y][x].b = rowImage->data[rowImage->height - haloThickness + y][x].b;
        }
      }
    }

    //Even ranks exchange bottom halo first
    if (evenRank && !bottomRank)
    {
      MPI_Sendrecv(bottomSendHalo->rawdata, rowImage->width * haloThickness, MPI_PIXEL, my_rank + 1, 0, bottomRecvHalo->rawdata, rowImage->width * haloThickness, MPI_PIXEL, my_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    //Odd ranks exchange top halo first
    else if (!evenRank)
    {
      MPI_Sendrecv(topSendHalo->rawdata, rowImage->width * haloThickness, MPI_PIXEL, my_rank - 1, 0, topRecvHalo->rawdata, rowImage->width * haloThickness, MPI_PIXEL, my_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    //Even ranks exchange top halo second
    if (evenRank && !topRank)
    {
      MPI_Sendrecv(topSendHalo->rawdata, rowImage->width * haloThickness, MPI_PIXEL, my_rank - 1, 0, topRecvHalo->rawdata, rowImage->width * haloThickness, MPI_PIXEL, my_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    //Odd ranks exchange bottom halo second
    else if (!evenRank && !bottomRank)
    {
      MPI_Sendrecv(bottomSendHalo->rawdata, rowImage->width * haloThickness, MPI_PIXEL, my_rank + 1, 0, bottomRecvHalo->rawdata, rowImage->width * haloThickness, MPI_PIXEL, my_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    //Applykernel is modified to include halos seperately
    applyKernel(rowBuffer->data,
                rowImage->data,
                rowImage->width,
                rowImage->height,
                kernels[kernelIndex],
                kernelDims[kernelIndex],
                kernelFactors[kernelIndex],
                topRecvHalo->data,
                bottomRecvHalo->data);
    swapImage(&rowBuffer, &rowImage);
  }

  //Gather all rows in original image-buffer
  MPI_Gatherv(rowImage->rawdata, counts[my_rank], MPI_PIXEL, image->rawdata, counts, offsets, MPI_PIXEL, 0, MPI_COMM_WORLD);

  freeBmpImage(rowImage);
  freeBmpImage(rowBuffer);

  if (my_rank == 0)
  {

    double spentTime = MPI_Wtime() - startTime;
    printf("Time spent: %.3f seconds\n", spentTime);

    //Write the image back to disk
    if (saveBmpImage(image, output) != 0)
    {
      fprintf(stderr, "Could not save output to '%s'!\n", output);
      freeBmpImage(image);
      goto error_exit;
    };
  }

graceful_exit:
  ret = 0;
error_exit:
  if (input)
    free(input);
  if (output)
    free(output);
  MPI_Finalize();
  return ret;
};
