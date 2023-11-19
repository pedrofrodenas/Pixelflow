#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(1, im.h, im.w);
    // TODO Fill this in
    int pixel_per_channel = im.w * im.h;
    for (size_t i = 0; i != pixel_per_channel; ++i)
    {
        gray.data[i] = 0.299 * im.data[i] + 0.587 * im.data[i + pixel_per_channel] + 0.114 * im.data[i + pixel_per_channel * 2];
    }
    return gray;
}

// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    // TODO Fill this in
    int rows = im.h;
    int cols = im.w;

    int channelOffset = rows * cols;

    float R, G, B, V, m, C, S, huep, H = 0;
    for (int i = 0; i != channelOffset; ++i)
    {
        R = im.data[i];
        G = im.data[i + channelOffset];
        B = im.data[i + channelOffset * 2];

        V = three_way_max(R, G, B);
        m = three_way_min(R, G, B);

        if (V == 0)
        {
            C = 0;
            S = 0;
        }
        else
        {
            C = V - m;
            S = C / V;
        }

        if (C == 0)
        {
            H = 0;
            im.data[i] = H;
            im.data[i + channelOffset] = S;
            im.data[i + channelOffset * 2] = V;
        }
        else
        {
            if (V == R)
            {
                huep = (G - B) / C;
            }
            else if (V == G)
            {
                huep = ((B - R) / C) + 2;
            }
            else if (V == B)
            {
                huep = ((R - G) / C) + 4;
            }
            if (huep < 0)
            {
                H = (huep / 6) + 1;
            }
            else
            {
                H = huep / 6;
            }
            im.data[i] = H;
            im.data[i + channelOffset] = S;
            im.data[i + channelOffset * 2] = V;
        }
    }
}

void hsv_to_rgb(image im)
{
    // TODO Fill this in
    // Assuming linear hue variation.
    int rows = im.h;
    int cols = im.w;

    int channelOffset = rows * cols;

    float R, G, B, V, m, C, S, huep, H = 0;
    for (int i = 0; i != channelOffset; ++i)
    {
        H = im.data[i];
        S = im.data[i + channelOffset];
        V = im.data[i + channelOffset * 2];

        // Get Chroma, or difference between max and min
        C = V * S;

        // We know that max is V and we know C so:
        m = V - C;

        // We want to know huep which vary between 0 and 6
        // to know to which corner of hexagon we are
        huep = H*6;

        // Now we want to get the equation of the line for each sector
        if ((huep >= 0) && (huep < 1)) 
        {
            // y = slope*x + n / G = C*mod1(huep) + m
            G = C*fmod(huep, 1) + m;
            R = V;
            B = m;
        }
        else if ((huep >= 1) && (huep < 2))
        {
            // y = slope*x + n 
            // -C due to negative slope
            R = -C*fmod(huep, 1) + V;
            G = V;
            B = m;
        }
        else if ((huep >= 2) && (huep < 3))
        {
            B = C*fmod(huep, 1) + m;
            G = V;
            R = m;
        }
        else if ((huep >= 3) && (huep < 4))
        {
            G = -C*fmod(huep, 1) + V;
            B = V;
            R = m;
        }
        else if ((huep >= 4) && (huep < 5))
        {
            R = C*fmod(huep, 1) + m;
            B = V;
            G = m;
        }
        else if ((huep >= 5) && (huep < 6))
        {
            B = -C*fmod(huep,1) + V;
            R = V;
            G = m;
        }
        else
        {
            R = 0;
            G = 0;
            B = 0;
        }
        im.data[i] = R;
        im.data[i + channelOffset] = G;
        im.data[i + channelOffset * 2] = B;
    }   
}