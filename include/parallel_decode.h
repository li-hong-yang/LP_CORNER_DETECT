#ifndef _PARALLEL_DECODE_H
#define _PARALLEL_DECODE_H

#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime_api.h"
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include "corner_detect.h"



void HandleError(cudaError_t err,
                        const char *file,
                        int line);
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

namespace nvinfer1
{
    
    int getThreadNum();
    __global__ void CalDetection(const float *input, float *output,int nums,float conf_thres,int outputElem,int c,int maxoutobject,float* priors,float* variances);



}

#endif