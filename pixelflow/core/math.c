#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

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