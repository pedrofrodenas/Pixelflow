#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
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

image make_box_filter(int w)
{
    // TODO
    image im = make_image(1,w,w);
    // Adds one to each image possition
    shift_image(im, 0, 1.0f);
    // Normalize values
    l1_normalize(im);
    return im;
}

image add_image(image a, image b)
{
    // TODO
    assert(a.h == b.h);
    assert(a.w == b.w);
    assert(a.c == b.c);

    image result = make_image(b.c, b.h, b.w);

    for (int c=0; c!=a.c; c++)
    {
        for (int y=0; y!=a.h; y++)
        {
            for (int x=0; x!=a.w; x++)
            {
                set_pixel(result, c, y, x, get_pixel(a, c, y, x)+get_pixel(b, c, y, x));
            }
        }
    }
    return result;
}

image sub_image(image a, image b)
{
    // TODO
    assert(a.h == b.h);
    assert(a.w == b.w);
    assert(a.c == b.c);

    image result = make_image(b.c, b.h, b.w);

    for (int c=0; c!=a.c; c++)
    {
        for (int y=0; y!=a.h; y++)
        {
            for (int x=0; x!=a.w; x++)
            {
                set_pixel(result, c, y, x, get_pixel(a, c, y, x)-get_pixel(b, c, y, x));
            }
        }
    }
    return result;
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

image colorize_sobel(image im)
{
    // TODO
    image filter = make_gaussian_filter(3);

    image outConv = convolve_image(im, filter, 1);

    image *magnitude = sobel_image(outConv);

    feature_normalize(magnitude[0]);
    feature_normalize(magnitude[1]);

    image output = make_image(im.c, im.h, im.w);

    for (int i=0; i!=im.h; i++)
    {
        for (int j=0; j!=im.w; j++)
        {
            set_pixel(output, 0, i, j, get_pixel(magnitude[1], 0, i, j));
            set_pixel(output, 1, i, j, get_pixel(magnitude[0], 0, i, j));
            set_pixel(output, 2, i, j, get_pixel(magnitude[0], 0, i, j));
        }
    }
    hsv_to_rgb(output);

    free_image(filter);
    free_image(outConv);
    
    for (int i=0; i<2; ++i)
    {
        free_image(magnitude[i]);
    }

    return output;
}
