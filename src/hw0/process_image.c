#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

float get_pixel(image im, int c, int h, int w)
{
    // TODO Fill this in
    int channels = im.c;
    int rows = im.h;
    int cols = im.w;

    // Clamp
    w = (w > 0) ? ((w < cols) ? w : (cols - 1)) : 0;
    h = (h > 0) ? ((h < rows) ? h : (rows - 1)) : 0;
    c = (c > 0) ? ((c < channels) ? c : (channels - 1)) : 0;

    float pixel = *(im.data + (rows * cols * c) + (cols * h) + w);
    return pixel;
}

void set_pixel(image im, int c, int h, int w, float v)
{
    // TODO Fill this in
    int rows = im.h;
    int cols = im.w;
    int channels = im.c;

    if (w < 0 || h < 0 || c < 0 || w >= cols || h >= rows || c >= channels)
    {
        return;
    }
    else
    {
        *(im.data + (rows * cols * c) + (cols * h + w)) = v;
        return;
    }
}

image copy_image(image im)
{
    image copy = make_image(im.c,  im.h, im.w);
    // TODO Fill this in
    size_t n_bytes = im.w * im.h * im.c * sizeof(float);
    memcpy(copy.data, im.data, n_bytes);
    return copy;
}

void shift_image(image im, int c, float v)
{
        // TODO Fill this in
    int pixel_per_channel = im.w * im.h;
    for (int i = 0; i < pixel_per_channel; ++i)
    {
        *(im.data + pixel_per_channel * c + i) = (*(im.data + pixel_per_channel * c + i)) + v;
    }
}

void clamp_image(image im)
{
    // TODO Fill this in
    int rows = im.h;
    int cols = im.w;
    int channels = im.c;

    int i = 0;
    int end = rows * cols * channels;

    while (i != end)
    {
        im.data[i] = (im.data[i] > 0) ? ((im.data[i] > 1) ? 1 : im.data[i]) : 0;
        i++;
    }
}

void scale_image(image im, int c, float v)
{
    float pixel=0;
    for (int y=0; y!=im.h; y++)
    {
        for (int x=0; x!=im.w; x++)
        {
            pixel = get_pixel(im, c, y, x);
            set_pixel(im, c, y, x, pixel*v);
        }
    }
}
