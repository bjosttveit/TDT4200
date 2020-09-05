#include <stdlib.h>
#include <stdio.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048
#define CHANNELS 3 // Three uchars per pixel (RGB)

//Inverts the color value of every channel in each pixel
void invert(uchar *array, int length)
{
	for (int i = 0; i < length; i++)
	{
		array[i] = 255 - array[i];
	}
}

//Swaps two color channels
void swapchannels(uchar *array, int length, int channel1, int channel2)
{
	for (int i = 0; i < length; i += 3)
	{
		uchar val = array[i + channel1];
		array[i + channel1] = array[i + channel2];
		array[i + channel2] = val;
	}
}

//Doubles the size of the image by copying every pixel four times
uchar *doubleSize(uchar *array, int width, int height, int channels)
{
	uchar *image = calloc(width * height * channels * 4, 1);

	for (int y = 0; y < height * 2; y++)
	{
		for (int x = 0; x < width * 2; x++)
		{
			for (int c = 0; c < channels; c++)
			{
				//Uses integer division to assign the same pixel in multiple locations, e.g. 2/2 = 3/2 = 1
				image[channels * (x + 2 * y * width) + c] = array[channels * (x / 2 + (y / 2) * width) + c];
			}
		}
	}

	return image;
}

int main()
{
	int length = XSIZE * YSIZE * CHANNELS;

	uchar *image = calloc(length, 1);
	readbmp("before.bmp", image);

	// Alter the image here
	invert(image, length);
	swapchannels(image, length, 0, 1);
	uchar *upscaledImage = doubleSize(image, XSIZE, YSIZE, CHANNELS);

	savebmp("after.bmp", upscaledImage, XSIZE * 2, YSIZE * 2);
	free(image);
	free(upscaledImage);
	return 0;
}