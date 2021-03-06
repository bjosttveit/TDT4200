#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

extern "C" {
    #include "libs/bitmap.h"
}

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__);  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
       if (code != cudaSuccess)
       {
                 fprintf(stderr,"GPUassert: %s %s %s %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
                       if (abort) exit(code);
                          
       }
}

#define GRID_SIZE 16
#define BLOCK_SIZE 128
#define TOTAL_THREAD_COUNT GRID_SIZE * BLOCK_SIZE

// Convolutional Filter Examples, each with dimension 3,
// gaussian filter with dimension 5

int sobelYFilter[] = {-1, -2, -1,
                       0,  0,  0,
                       1,  2,  1};

int sobelXFilter[] = {-1, -0, 1,
                      -2,  0, 2,
                      -1,  0, 1};

int laplacian1Filter[] = { -1,  -4,  -1,
                           -4,  20,  -4,
                           -1,  -4,  -1};

int laplacian2Filter[] = { 0,  1,  0,
                           1, -4,  1,
                           0,  1,  0};

int laplacian3Filter[] = { -1,  -1,  -1,
                           -1,   8,  -1,
                           -1,  -1,  -1};

int gaussianFilter[] = { 1,  4,  6,  4, 1,
                         4, 16, 24, 16, 4,
                         6, 24, 36, 24, 6,
                         4, 16, 24, 16, 4,
                         1,  4,  6,  4, 1 };

const char* filterNames[]       = { "SobelY",     "SobelX",     "Laplacian 1",    "Laplacian 2",    "Laplacian 3",    "Gaussian"     };
int* const filters[]            = { sobelYFilter, sobelXFilter, laplacian1Filter, laplacian2Filter, laplacian3Filter, gaussianFilter };
unsigned int const filterDims[] = { 3,            3,            3,                3,                3,                5              };
float const filterFactors[]     = { 1.0,          1.0,          1.0,              1.0,              1.0,              1.0 / 256.0    };

int const maxFilterIndex = sizeof(filterDims) / sizeof(unsigned int);

void cleanup(char** input, char** output) {
    if (*input)
        free(*input);
    if (*output)
        free(*output);
}

void graceful_exit(char** input, char** output) {
    cleanup(input, output);
    exit(0);
}

void error_exit(char** input, char** output) {
    cleanup(input, output);
    exit(1);
}

// Helper function to swap bmpImageChannel pointers

void swapImageRawdata(pixel **one, pixel **two) {
  pixel *helper = *two;
  *two = *one;
  *one = helper;
}

void swapImage(bmpImage **one, bmpImage **two) {
  bmpImage *helper = *two;
  *two = *one;
  *one = helper;
}

// Apply convolutional filter on image data using gpu
__global__ void applyFilterDevice(pixel *out, pixel *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor) {
  unsigned int const filterCenter = (filterDim / 2);
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int threadNumber = threadIdx.x + blockDim.x * threadIdx.y;
  unsigned int blockSize = blockDim.x * blockDim.y;
  unsigned int bufferWidth = (blockDim.x + 2*filterCenter);

  extern __shared__ int s[];

  //Copy filter to shared memory
  int filterLength = filterDim * filterDim;
  int *f = s;
  for (unsigned int i = threadNumber; i < filterLength; i += blockSize) {
    f[i] = filter[i];
  }

  //Copy pixels to shared memory
  //It takes the filtersize into account
  //and includes enough pixels around the block
  pixel *buffer = (pixel*)&f[filterLength];
  int startX = blockIdx.x * blockDim.x;
  int startY = blockIdx.y * blockDim.y;
  for (int by = threadIdx.y; by < blockDim.y + 2*filterCenter; by += blockDim.y) {
    for (int bx = threadIdx.x; bx < blockDim.x + 2*filterCenter; bx += blockDim.x) {
      int yy = startY + by - filterCenter;
      int xx = startX + bx - filterCenter;
      if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height) {
        buffer[by * bufferWidth + bx] = in[yy*width + xx];
      }
    }
  }

  __syncthreads();


  if (x < width && y < height) {
    int ar = 0, ag = 0, ab = 0;
    for (unsigned int ky = 0; ky < filterDim; ky++) {
      int nky = filterDim - 1 - ky;
      for (unsigned int kx = 0; kx < filterDim; kx++) {
        int nkx = filterDim - 1 - kx;

        //xx and yy is the position of the pixel on the actual image
        int yy = y + (ky - filterCenter);
        int xx = x + (kx - filterCenter);
        //bx and by is the position of the same pixel in the shared memory buffer
        int by = yy - startY + filterCenter;
        int bx = xx - startX + filterCenter;
        if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height) {
          ar += buffer[by*bufferWidth + bx].r * f[nky * filterDim + nkx];
          ag += buffer[by*bufferWidth + bx].g * f[nky * filterDim + nkx];
          ab += buffer[by*bufferWidth + bx].b * f[nky * filterDim + nkx];
        }
      }
    }

      ar *= filterFactor;
      ag *= filterFactor;
      ab *= filterFactor;
      
      ar = (ar < 0) ? 0 : ar;
      ag = (ag < 0) ? 0 : ag;
      ab = (ab < 0) ? 0 : ab;

      out[y*width +x].r = (ar > 255) ? 255 : ar;
      out[y*width +x].g = (ag > 255) ? 255 : ag;
      out[y*width +x].b = (ab > 255) ? 255 : ab;
  }
}

void help(char const *exec, char const opt, char const *optarg) {
    FILE *out = stdout;
    if (opt != 0) {
        out = stderr;
        if (optarg) {
            fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
        } else {
            fprintf(out, "Invalid parameter - %c\n", opt);
        }
    }
    fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
    fprintf(out, "\n");
    fprintf(out, "Options:\n");
    fprintf(out, "  -k, --filter     <filter>        filter index (0<=x<=%u) (2)\n", maxFilterIndex -1);
    fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

    fprintf(out, "\n");
    fprintf(out, "Example: %s before.bmp after.bmp -i 10000\n", exec);
}

size_t blockSizeToDynamicSMemSize(int blockSize) {
  //Assumes that the blockdimensions are square, 3x3 filter is assumed in this case
  size_t sharedMemSize = 9*sizeof(int) + (blockSize + 4 * sqrt(blockSize) + 4) * sizeof(pixel);
  return sharedMemSize;
}


int main(int argc, char **argv) {
  /*
    Parameter parsing, don't change this!
   */
  unsigned int iterations = 1;
  char *output = NULL;
  char *input = NULL;
  unsigned int filterIndex = 2;

  static struct option const long_options[] =  {
      {"help",       no_argument,       0, 'h'},
      {"filter",     required_argument, 0, 'k'},
      {"iterations", required_argument, 0, 'i'},
      {0, 0, 0, 0}
  };

  static char const * short_options = "hk:i:";
  {
    char *endptr;
    int c;
    int parse;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
      switch (c) {
      case 'h':
        help(argv[0],0, NULL);
        graceful_exit(&input,&output);
      case 'k':
        parse = strtol(optarg, &endptr, 10);
        if (endptr == optarg || parse < 0 || parse >= maxFilterIndex) {
          help(argv[0], c, optarg);
          error_exit(&input,&output);
        }
        filterIndex = (unsigned int) parse;
        break;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg) {
          help(argv[0], c, optarg);
          error_exit(&input,&output);
        }
        break;
      default:
        abort();
      }
    }
  }

  if (argc <= (optind+1)) {
    help(argv[0],' ',"Not enough arugments");
    error_exit(&input,&output);
  }

  unsigned int arglen = strlen(argv[optind]);
  input = (char*)calloc(arglen + 1, sizeof(char));
  strncpy(input, argv[optind], arglen);
  optind++;

  arglen = strlen(argv[optind]);
  output = (char*)calloc(arglen + 1, sizeof(char));
  strncpy(output, argv[optind], arglen);
  optind++;

  /*
    End of Parameter parsing!
   */


  /*
    Create the BMP image and load it from disk.
   */
  bmpImage *image = newBmpImage(0,0);
  if (image == NULL) {
    fprintf(stderr, "Could not allocate new image!\n");
    error_exit(&input,&output);
  }

  if (loadBmpImage(image, input) != 0) {
    fprintf(stderr, "Could not load bmp image '%s'!\n", input);
    freeBmpImage(image);
    error_exit(&input,&output);
  }

  printf("Apply filter '%s' on image with %u x %u pixels for %u iterations\n", filterNames[filterIndex], image->width, image->height, iterations);


  // TODO: implement time measurement from here
  //struct timespec start_time, end_time;
  //clock_gettime(CLOCK_MONOTONIC, &start_time);

  // Here we do the actual computation!
  // image->data is a 2-dimensional array of pixel which is accessed row first ([y][x])
  // image->rawdata is a 1-dimensional array of pixel containing the same data as image->data
  // each pixel is a struct of 3 unsigned char for the red, blue and green colour channel
  bmpImage *processImage = newBmpImage(image->width, image->height);

  // TODO: Cuda malloc and memcpy the rawdata from the images, from host side to device side
  pixel *devicePixelsIn;
  pixel *devicePixelsOut;
  int *filter;
  cudaMalloc((void**)&devicePixelsIn, image->width * image->height * sizeof(pixel));
  cudaMalloc((void**)&devicePixelsOut, image->width * image->height * sizeof(pixel));
  cudaMalloc((void**)&filter, filterDims[filterIndex]*filterDims[filterIndex]*sizeof(int));
  cudaMemcpy(devicePixelsIn, image->rawdata, image->width * image->height * sizeof(pixel), cudaMemcpyHostToDevice);
  cudaMemcpy(devicePixelsOut, processImage->rawdata, image->width * image->height * sizeof(pixel), cudaMemcpyHostToDevice);
  cudaMemcpy(filter, filters[filterIndex], filterDims[filterIndex]*filterDims[filterIndex]*sizeof(int), cudaMemcpyHostToDevice);

  // TODO: Define the gridSize and blockSize, e.g. using dim3 (see Section 2.2. in CUDA Programming Guide)
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks((image->width + threadsPerBlock.x - 1) / threadsPerBlock.x, (image->height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  //Specify the amount of shared memory for filter and pixel blocks
  int filterLength = filterDims[filterIndex]*filterDims[filterIndex];
  int filterCenter = filterDims[filterIndex] / 2;
  int blockBufferLength = (threadsPerBlock.x + 2*filterCenter)*(threadsPerBlock.y + 2*filterCenter);
  size_t sharedMemSize = filterLength*sizeof(int) + blockBufferLength*sizeof(pixel);

  //Occupancy max potential block size
  int blockSize;
  int minGridSize;
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, applyFilterDevice, blockSizeToDynamicSMemSize, 0);

  // TODO: Intialize and start CUDA timer
  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  cudaError_t error = cudaPeekAtLastError();
  if (error) {
      fprintf(stderr, "1: A CUDA error has occurred: %s\n", cudaGetErrorString(error));
  }

  for (unsigned int i = 0; i < iterations; i ++) {
      // TODO: Implement kernel call instead of serial implementation
    applyFilterDevice<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        devicePixelsOut,
		    devicePixelsIn,
		    image->width,
		    image->height,
		    filter,
		    filterDims[filterIndex],
		    filterFactors[filterIndex]
    );

    error = cudaPeekAtLastError();
    if (error) {
        fprintf(stderr, "KERNEL: A CUDA error has occurred: %s\n", cudaGetErrorString(error));
    }

    //swapImage(&processImage, &image);
    swapImageRawdata(&devicePixelsOut, &devicePixelsIn);

    error = cudaPeekAtLastError();
    if (error) {
        fprintf(stderr, "SWAP: A CUDA error has occurred: %s\n", cudaGetErrorString(error));
    }
  }

  error = cudaPeekAtLastError();
  if (error) {
      fprintf(stderr, "2: A CUDA error has occurred: %s\n", cudaGetErrorString(error));
  }

  // TODO: Stop CUDA timer
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &end_time);

  // TODO: Copy back rawdata from images
  cudaMemcpy(image->rawdata, devicePixelsIn, image->width * image->height * sizeof(pixel), cudaMemcpyDeviceToHost);

  
  error = cudaPeekAtLastError();
  if (error) {
      fprintf(stderr, "3: A CUDA error has occurred: %s\n", cudaGetErrorString(error));
  }

  // TODO: Calculate and print elapsed time
  //cudaDeviceSynchronize();
  //clock_gettime(CLOCK_MONOTONIC, &end_time);
  double spentTime = ((double) (end_time.tv_sec - start_time.tv_sec)) + ((double) (end_time.tv_nsec - start_time.tv_nsec)) * 1e-9;
  printf("Time spent: %.5f seconds\n", spentTime);

  //Occupancy calculation taken directly from: https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
  int maxActiveBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, applyFilterDevice, blockSize, sharedMemSize);

  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);

  float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / (float)(props.maxThreadsPerMultiProcessor / props.warpSize);
  printf("Launched blocks of size %d. Theoretical occupancy: %f\n", blockSize, occupancy);

  freeBmpImage(processImage);
  //Write the image back to disk
  if (saveBmpImage(image, output) != 0) {
    fprintf(stderr, "Could not save output to '%s'!\n", output);
    freeBmpImage(image);
    error_exit(&input,&output);
  };

  graceful_exit(&input,&output);
};
