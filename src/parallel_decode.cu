#ifndef _PARALLEL_DECODE_H
#define _PARALLEL_DECODE_H
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime_api.h"
#include <device_launch_parameters.h>
#include <cmath>
#include "parallel_decode.h"
#include "corner_detect.h"





namespace nvinfer1
{

    
static void HandleError(cudaError_t err,
                        const char *file,
                        int line)
                        {
                            if(err != cudaSuccess)
                            {
                                printf("%s in %s at line %d\n",
                                cudaGetErrorString(err),
                                file, line);
                                exit(EXIT_FAILURE);
                            }
                        }
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))



__global__ void CalDetection(const float *input, float *output,int nums,float conf_thres,int outputElem,int c,int maxoutobject,float* priors,float* variances)
{   //  noElements: feature_map_h*feature_map_w  netheight:608 maxoutobject:1000 classes:80 outputElem:6*1000+1
    // 每个线程处理一个featur_map中的一个点的数据

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= nums) return;

    if (input[idx*c + 5] <= conf_thres) return;

    float *res_count = output; // 预留 1000个位置
    int count = (int)atomicAdd(res_count, 1); // cuda原子操作 计算(old + val)
    if (count >= maxoutobject) return;
    char* data = (char *)res_count + sizeof(float) + count * sizeof(Yolo::Detection);
    Yolo::Detection* det = (Yolo::Detection*)(data);

    float x = priors[idx*c+0] + input[idx*c+0] * variances[0] * priors[idx*c+2];
    float y = priors[idx*c+1] + input[idx*c+1] * variances[0] * priors[idx*c+3];
    float w = priors[idx*c+2] + exp(input[idx*c+2] * variances[1]); // cuda exp 计算
    float h = priors[idx*c+3] + exp(input[idx*c+3] * variances[1]);

    det->bbox[0] = 416*(x-w/2.0);
    det->bbox[1] = 416*(y-h/2.0);
    det->bbox[2] = 416*(x+w/2.0);
    det->bbox[3] = 416*(y+h/2.0);
    det->conf = input[idx*c + 5];

    det->corner[0] = (priors[idx*c+0] + input[idx*c+0] * variances[0] * priors[idx*c+2])*416;
    det->corner[1] = (priors[idx*c+1] + input[idx*c+1] * variances[0] * priors[idx*c+3])*416;

    det->corner[2] = (priors[idx*c+0] + input[idx*c+2] * variances[0] * priors[idx*c+2])*416;
    det->corner[3] = (priors[idx*c+1] + input[idx*c+3] * variances[0] * priors[idx*c+3])*416;

    det->corner[4] = (priors[idx*c+0] + input[idx*c+4] * variances[0] * priors[idx*c+2])*416;
    det->corner[5] = (priors[idx*c+1] + input[idx*c+5] * variances[0] * priors[idx*c+3])*416;

    det->corner[6] = (priors[idx*c+0] + input[idx*c+6] * variances[0] * priors[idx*c+2])*416;
    det->corner[7] = (priors[idx*c+1] + input[idx*c+7] * variances[0] * priors[idx*c+3])*416;
};


int getThreadNum()
{
    cudaDeviceProp prop;
    int count;

    HANDLE_ERROR(cudaGetDeviceCount(&count));
    printf("Gpu Num: %d\n",count);
    HANDLE_ERROR(cudaGetDeviceProperties(&prop,0));
    printf("Max Thread Num: %d\n",prop.maxThreadsPerBlock);
    printf("Max Grid Dimensions: %d %d %d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);

    return prop.maxThreadsPerBlock;

}

}


#endif