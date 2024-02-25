#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "pixelflow/legacy/image.h"
#define TWOPI 6.2831853

void l1_normalize(image im)
{
    // TODO
    // Assert 1 channel image
    assert(im.c == 1);

    float cumsum = 0.f;
    int size = im.h*im.w;

    for (int i=0; i<size; i++)
    {
        cumsum += im.data[i];
    }

    if (cumsum == 0)
    {
        printf("Division by 0 in l1_normalize, all values in image are 0's \n");
    }
    scale_image(im, 0, 1.f/cumsum);
}

void feature_normalize(image im)
{
    // TODO
    float min, max, aux;

    min = INFINITY;
    max = -1;

    for (int c=0; c!=im.c; c++)
    {
        for (int y=0; y!=im.h; y++)
        {
            for (int x=0; x!=im.w; x++)
            {
                aux = get_pixel(im, c, y, x);
                if (aux < min)
                {
                    min = aux;
                }
                if (aux > max)
                {
                    max = aux;
                }
            }
        }
    }

    for (int c=0; c!=im.c; c++)
    {
        for (int y=0; y!=im.h; y++)
        {
            for (int x=0; x!=im.w; x++)
            {
                if ((max-min)==0)
                {
                    set_pixel(im, c, y, x, 0);
                }
                else
                {   float pixel = get_pixel(im, c, y, x);
                    set_pixel(im, c, y, x, (pixel-min)/(max-min));
                }
            }
        }
    }
}