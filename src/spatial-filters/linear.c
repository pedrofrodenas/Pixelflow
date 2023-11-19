#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

image convolve_image(image im, image filter, int preserve)
{
    // TODO
    assert(im.c == filter.c || filter.c == 1);

    // Make resulting image
    int outChannels = (preserve) ? im.c : 1;

    image imgRes = make_image(outChannels, im.h, im.w);
    
    // Padding amount
    int Padx = floor(filter.w/2.);
    int Pady = floor(filter.h/2.);

    // Pixels to get from image and filter
    float filPix, imPix = 0.f;

    for (int y=0; y!=im.h; y++)
    {
        for (int x=0; x!=im.w; x++)
        {   
            if (preserve == 1)
            {
                for (int ch=0; ch!=im.c; ch++)
                {
                    float dstPix = 0.;
                    for (int Fy = 0; Fy!=filter.h; Fy++)
                    {
                        for (int Fx=0; Fx!=filter.w; Fx++)
                        {
                            filPix = get_pixel(filter, 0, Fy, Fx);
                            imPix = get_pixel(im, ch,  y+Fy-Pady, x+Fx-Padx);
                            dstPix += (imPix*filPix);
                        }
                    }
                    set_pixel(imgRes, ch, y, x, dstPix);
                }
            }
            else
            {
                float dstPix = 0.;
                for (int Fy = 0; Fy!=filter.h; Fy++)
                {
                    for (int Fx=0; Fx!=filter.w; Fx++)
                    {
                        for (int ch=0; ch!=im.c; ch++)
                        {
                            imPix = get_pixel(im, ch, y+Fy-Pady, x+Fx-Padx);
                            if (im.c == filter.c)
                            {
                                filPix = get_pixel(filter, ch, Fy, Fx);
                            }
                            else
                            {
                                filPix = get_pixel(filter, 0, Fy, Fx);
                            }  
                            dstPix += (imPix*filPix);   
                        }
                    }
                }
                set_pixel(imgRes, 0, y, x, dstPix);
            }
        }
    }
    return imgRes;
}

image make_highpass_filter()
{
    image highPassFilter = make_image(1,3,3);
    // First row values
    set_pixel(highPassFilter, 0, 0, 1, -1);
    // Second row values
    set_pixel(highPassFilter, 0, 1, 0, -1);
    set_pixel(highPassFilter, 0, 1, 1, 4);
    set_pixel(highPassFilter, 0, 1, 2, -1);
    // Third row values
    set_pixel(highPassFilter, 0, 2, 1, -1);
    
    return highPassFilter;
}

image make_sharpen_filter()
{
    // TODO
    image sharpenFilter = make_image(1,3,3);
    // First row values
    set_pixel(sharpenFilter, 0, 0, 1, -1);
    // Second row values
    set_pixel(sharpenFilter, 0, 1, 0, -1);
    set_pixel(sharpenFilter, 0, 1, 1, 5);
    set_pixel(sharpenFilter, 0, 1, 2, -1);
    // Third row values
    set_pixel(sharpenFilter, 0, 2, 1, -1);
    
    return sharpenFilter;
}

image make_emboss_filter()
{
    // TODO
    image embossFilter = make_image(1,3,3);
    // First row values
    set_pixel(embossFilter, 0, 0, 0, -2);
    set_pixel(embossFilter, 0, 0, 1, -1);
    // Second row values
    set_pixel(embossFilter, 0, 1, 0, -1);
    set_pixel(embossFilter, 0, 1, 1, 1);
    set_pixel(embossFilter, 0, 1, 2, 1);
    // Third row values
    set_pixel(embossFilter, 0, 2, 1, 1);
    set_pixel(embossFilter, 0, 2, 2, 2);

    return embossFilter;
}

// Question 2.2.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Answer: TODO

// Question 2.2.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: TODO

image make_gaussian_filter(float sigma)
{
    // TODO
    // sigma * 6
    int gaussianSize = (int)ceil(sigma*6);
    // odd sigma* 6
    gaussianSize = (fmod(gaussianSize,2) == 0) ? gaussianSize+1: gaussianSize;

    image gaussian = make_image(1, gaussianSize,gaussianSize);

    // Base and exponenet for sigma calculation
    float base, exponent = 0.F;

    // Shifting x and y to center filter grid on 0
    int shiftingXY = floor(gaussianSize/2);

    for (int y=0; y!=gaussian.h; y++)
    {
        for (int x=0; x!=gaussian.w; x++)
        {
            base = 1.0f / (TWOPI*powf(sigma, 2.f));
            exponent = expf(-(powf(x-shiftingXY,2)+powf(y-shiftingXY,2))/(2*powf(sigma,2)));
            set_pixel(gaussian, 0, y, x, base*exponent);
        }
    }

    l1_normalize(gaussian);

    return gaussian;
}

image make_gx_filter()
{
    // TODO
    image sobelXFilter = make_image(1, 3, 3);
    // First row values
    set_pixel(sobelXFilter, 0, 0, 0, -1);
    set_pixel(sobelXFilter, 0, 0, 2, 1);
    // Second row values
    set_pixel(sobelXFilter, 0, 1, 0, -2);
    set_pixel(sobelXFilter, 0, 1, 2, 2);
    // Third row values
    set_pixel(sobelXFilter, 0, 2, 0, -1);
    set_pixel(sobelXFilter, 0, 2, 2, 1);

    return sobelXFilter;
}

image make_gy_filter()
{
    // TODO
    image sobelYFilter = make_image(1, 3, 3);
    // First row values
    set_pixel(sobelYFilter, 0, 0, 0, -1);
    set_pixel(sobelYFilter, 0, 0, 1, -2);
    set_pixel(sobelYFilter, 0, 0, 2, -1);
    // Second row values
    // Third row values
    set_pixel(sobelYFilter, 0, 2, 0, 1);
    set_pixel(sobelYFilter, 0, 2, 1, 2);
    set_pixel(sobelYFilter, 0, 2, 2, 1);

    return sobelYFilter;
}

image *sobel_image(image im)
{
    image *imgarray = calloc(2, sizeof(image));

    image sobelxF = make_gx_filter();
    image sobelyF = make_gy_filter();

    image sobelXout = convolve_image(im, sobelxF, 0);
    image sobelYout = convolve_image(im, sobelyF, 0);

    imgarray[0] = make_image(sobelXout.c, sobelXout.h, sobelXout.w);
    imgarray[1] = make_image(sobelXout.c, sobelXout.h, sobelXout.w);

    float Gx, Gy;
    for (int y = 0; y!= sobelXout.h; y++)
    {
        for (int x=0; x!=sobelXout.w; x++)
        {
                Gx = get_pixel(sobelXout, 0, y, x);
                Gy = get_pixel(sobelYout, 0, y, x);
                set_pixel(imgarray[0], 0, y, x, sqrt(pow(Gx, 2)+pow(Gy, 2)));
                set_pixel(imgarray[1], 0, y, x,  atan2f(Gy, Gx));
        }
    }
    free_image(sobelxF);
    free_image(sobelyF);
    free_image(sobelXout);
    free_image(sobelYout);

    return imgarray;
}