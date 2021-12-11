#ifndef _PARA_DECODE_H
#define _PARA_DECODE_H

#include <vector>
#include <string>
#include "NvInfer.h"
#include "corner_detect.h"
#include <device_launch_parameters.h>


namespace nvinfer1
{
    static void HandleError(cudaError_t err,
                        const char *file,
                        int line);
    int getThreadNum();
    __global__ void CalDetection(const float *input, float *output,int nums,float conf_thres,int outputElem,int c,int maxoutobject,float* priors,float* variances);



}










#endif