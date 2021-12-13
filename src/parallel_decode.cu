#include "parallel_decode.h"




void HandleError(cudaError_t err,
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


namespace nvinfer1
{


__global__ void CalDetection(const float *input, float *output,int nums,float conf_thres,int outputElem,int c,int maxoutobject,float* priors,float* variances)
{   

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= nums) return;

    if (input[idx*c + 5] <= conf_thres) return;

    float *res_count = output+maxoutobject*13; // 预留 1000个位置
    int count = (int)atomicAdd(res_count, 1); // cuda原子操作 计算(old + val)
    
    if (count >= maxoutobject) return;
    char* data = (char *)output + count * sizeof(Yolo::Detection);
    Yolo::Detection* det = (Yolo::Detection*)(data);

    float x = priors[idx*4+0] + input[idx*c+0] * variances[0] * priors[idx*4+2];
    float y = priors[idx*4+1] + input[idx*c+1] * variances[0] * priors[idx*4+3];
    float w = priors[idx*4+2] * expf(input[idx*c+2] * variances[1]); // cuda exp 计算
    float h = priors[idx*4+3] * expf(input[idx*c+3] * variances[1]);




    det->bbox[0] = 416*(x-w/2.0);
    det->bbox[1] = 416*(y-h/2.0);
    det->bbox[2] = 416*(x+w/2.0);
    det->bbox[3] = 416*(y+h/2.0);
    det->conf = input[idx*c + 5];

    det->corner[0] = (priors[idx*4+0] + input[idx*c+6] * variances[0] * priors[idx*4+2])*416;
    det->corner[1] = (priors[idx*4+1] + input[idx*c+7] * variances[0] * priors[idx*4+3])*416;

    det->corner[2] = (priors[idx*4+0] + input[idx*c+8] * variances[0] * priors[idx*4+2])*416;
    det->corner[3] = (priors[idx*4+1] + input[idx*c+9] * variances[0] * priors[idx*4+3])*416;

    det->corner[4] = (priors[idx*4+0] + input[idx*c+10] * variances[0] * priors[idx*4+2])*416;
    det->corner[5] = (priors[idx*4+1] + input[idx*c+11] * variances[0] * priors[idx*4+3])*416;

    det->corner[6] = (priors[idx*4+0] + input[idx*c+12] * variances[0] * priors[idx*4+2])*416;
    det->corner[7] = (priors[idx*4+1] + input[idx*c+13] * variances[0] * priors[idx*4+3])*416;

    if (idx == 0)
    {
        printf("==input==\n");
        printf("%f\n",input[idx*c+0]);
        printf("%f\n",input[idx*c+1]);
        printf("%f\n",input[idx*c+2]);
        printf("%f\n",input[idx*c+3]);

        printf("==priors==\n");
        printf("%f\n",priors[idx*4+0]);
        printf("%f\n",priors[idx*4+1]);
        printf("%f\n",priors[idx*4+2]);
        printf("%f\n",priors[idx*4+3]);

        printf("==bbox==\n");
        printf("%f\n",det->bbox[0]);
        printf("%f\n",det->bbox[1]);
        printf("%f\n",det->bbox[2]);
        printf("%f\n",det->bbox[3]);


        printf("==corner==\n");
        printf("%f\n",det->corner[0]);
        printf("%f\n",det->corner[1]);
        printf("%f\n",det->corner[2]);
        printf("%f\n",det->corner[3]);
        printf("%f\n",det->corner[4]);
        printf("%f\n",det->corner[5]);
        printf("%f\n",det->corner[6]);
        printf("%f\n",det->corner[7]);

        printf("==variances==\n");
        printf("%f\n", variances[0]);
        printf("%f\n", variances[1]);

    }
};


int getThreadNum()
{
    cudaDeviceProp prop;
    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop,0));
    // printf("Max Thread Num: %d\n",prop.maxThreadsPerBlock);
    // printf("Max Grid Dimensions: %d %d %d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);

    return prop.maxThreadsPerBlock;

}

}


