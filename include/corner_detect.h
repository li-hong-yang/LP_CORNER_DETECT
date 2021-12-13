#ifndef _CORNER_DETECT_H_
#define _CORNER_DETECT_H_

#include <string>
#include "opencv2/opencv.hpp"
#include "NvInfer.h"
#include <chrono>
#include "cuda_utils.h"
#include "logging.h"
#include <vector>
#include "parallel_decode.h"
using namespace std;





namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 1;
    static constexpr int INPUT_H = 416;  // yolov5's input height and width must be divisible by 32.
    static constexpr int INPUT_W = 416;

    static constexpr int LOCATIONS = 4;
    static constexpr int CORNERS = 8;
    struct alignas(float) Detection {
        //center_x center_y w h
        float bbox[LOCATIONS];
        float corner[CORNERS];
        float conf;  // bbox_conf * cls_conf
    };
}


class CornerDetect
{
public:
    CornerDetect(const std::string & engine_name,string& bbox);
    ~CornerDetect(); 
    void preprocess(string& img_name);
    void postprocess(string& img_name,float conf_thresh,float nms_thresh);
    void infer();

private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;
    void* buffers[2];           // context input and output
    float* data;                // context input
    float* output_buffer;       // context output
    float* decode_bbox;         // bbox_priors
    float* output;              // output decode specify-format Yolo::Detection

    float* gpu_input;           // cuda kenel input 
    float* gpu_output;          // cuda kenel output
    float* gpu_priors;          // cuda kenel priors 
    float* gpu_variances;       // cuda kenel variances 

    const int batch_size = 1;
    const int input_c = 3;       // 通道数
    const int input_w = 416;     // 特征向量维数w
    const int input_h = 416;     // 特征向量维数h
    const int device = 0;
    const int output_size = 7098 *14;

    float variances[2] = {0.1,0.2}; // 
    int nums = 7098;
    int c = 14;
    int maxoutobject = 7098;
    float conf_thres = 0.3;
    int det_size = 14;
    
};

#endif // _CORNER_DETECT_H_
