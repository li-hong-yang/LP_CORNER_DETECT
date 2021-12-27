#include <iostream>
#include <chrono>
#include "cuda_utils.h"
#include "logging.h"
#include "corner_detect.h"
#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>
#include "decode_utils.h"

// test odasf

static Logger gLogger;

using namespace nvinfer1;
using namespace std;

// 加载模型，分配显存和内存
CornerDetect::CornerDetect(const std::string & engine_name,string& bbox)
{
    // select device
    cudaSetDevice(device);

    // load TRT-ENGINE
    std::ifstream file(engine_name, std::ios::binary);
    assert(file.good() == true);
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // build trt context
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);

    // set input output idx
    int inputIndex, outputIndex;
    for (int bi = 0; bi < engine->getNbBindings(); bi++)
    {
        if (engine->bindingIsInput(bi) == true)
        {
            inputIndex = bi;
        }
        else
        {
            outputIndex = bi;
        }
    }
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batch_size * input_c * input_w * input_h *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batch_size * output_size * sizeof(float)));
    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));
    data = new float[batch_size * input_c * input_w * input_h];
    assert(data != nullptr);
    output_buffer = new float[batch_size * output_size];  
    assert(output_buffer != nullptr);  


    // read bbox_priors
    decode_bbox = new float[7098*4];
    assert(decode_bbox != nullptr);
    std::ifstream in(bbox, std::ios::in | std::ios::binary); 
    in.read((char *)decode_bbox, sizeof(float)*7098*4);
    in.close();


    // malloc related memory 
    output = new float[maxoutobject*13+1];
    HANDLE_ERROR(cudaMalloc((void**)&gpu_input,   sizeof(float)*batch_size * output_size));
    HANDLE_ERROR(cudaMalloc((void**)&gpu_output, sizeof(float)*(maxoutobject*13+1)));
    HANDLE_ERROR(cudaMalloc((void**)&gpu_priors,   sizeof(float)*nums*4));
    HANDLE_ERROR(cudaMalloc((void**)&gpu_variances,   sizeof(float)*2));
    HANDLE_ERROR(cudaMemcpy(gpu_variances,variances,sizeof(float)*2,cudaMemcpyHostToDevice));
}


void CornerDetect::preprocess(string& img_name)
{
    cv::Mat img = cv::imread(img_name);
    cv::Mat out;
    cv::resize(img,out,cv::Size(input_w,input_h));
    int i = 0;
    for (int row = 0; row < input_h; ++row) {
        uchar* uc_pixel = out.data + row * out.step;
        for (int col = 0; col < input_w; ++col) {
            data[0 * 3 * input_h * input_w + i] = (float)uc_pixel[0] - 104.0;
            data[0 * 3 * input_h * input_w + i + input_h * input_w] = (float)uc_pixel[1] -117.0;
            data[0 * 3 * input_h * input_w + i + 2 * input_h * input_w] = (float)uc_pixel[2] -123.0;
            uc_pixel += 3;
            ++i;
        }
    }
    cout << "pre_deal_done" << endl;
    

}


void CornerDetect::postprocess(string& img_name,float conf_thresh,float nms_thresh)
{

   
    
    cv::Mat img = cv::imread(img_name);
    cv::Mat imgraw;
    cv::resize(img,imgraw,cv::Size(416,416));
    int imgh = img.rows;
    int imgw = img.cols;

    float fx = imgw/416.0;
    float fy = imgh/416.0;

    // cuda kernel infer 
    HANDLE_ERROR(cudaMemcpy(gpu_input,output_buffer,sizeof(float)*batch_size * output_size,cudaMemcpyHostToDevice));  
    HANDLE_ERROR(cudaMemcpy(gpu_priors,decode_bbox,sizeof(float)*nums*4,cudaMemcpyHostToDevice));
    int threadNum = getThreadNum();
    int blockNum = (nums -0.5)/threadNum +1;
    CalDetection <<<blockNum,threadNum>>>(gpu_input,gpu_output,nums,conf_thres,nums,c,maxoutobject,gpu_priors,gpu_variances);
    HANDLE_ERROR(cudaMemcpy(output,gpu_output,sizeof(float)*(maxoutobject*13+1),cudaMemcpyDeviceToHost));

    // parse results acording Yolo::Detection format
    std::vector<Yolo::Detection> res;
    std::vector<Yolo::Detection> dst;
    printf("Fill conf condition num is:%d\n",int(output[maxoutobject*13]));

    for (int i=0;i<int(output[maxoutobject*13]);++i)
    {
        Yolo::Detection tmp;
        memcpy(tmp.bbox,&output[i*13+0],sizeof(float)*4);
        memcpy(&tmp.corner,&output[i*13+4],sizeof(float)*8);// 解析时注意顺序
        memcpy(&tmp.conf,&output[i*13+12],sizeof(float)*1);
        
        dst.push_back(tmp);     
    }
    
    // nms
    std::sort(dst.begin(), dst.end(), cmp);
    for (size_t m = 0; m < dst.size(); ++m) {
        auto& item = dst[m];
        res.push_back(item);
        for (size_t n = m + 1; n < dst.size(); ++n) {
            if (iou(item.bbox, dst[n].bbox) > nms_thresh) {
                dst.erase(dst.begin() + n);
                --n;
            }
        }
    }
   
    //  save img with bbox
    for (size_t j = 0; j < 1; j++) {
        cv::Rect r = get_rect(res[j].bbox,fx,fy);
        cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(img, "lp", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
    cv::imwrite("lp.jpg", img);

    //  save perspective_img
    cv::Mat pers_img;
    cv::Mat out_img;
    pers_img = get_perspective_mat(res[0].corner);
    cv::warpPerspective(imgraw,out_img,pers_img,cv::Size(94,24));
    cv::imwrite("pers_lp.jpg", out_img);
    cout << "save_done" << endl;

}

void CornerDetect::infer()
{
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], data,  batch_size * input_c * input_w * input_h * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueueV2(buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output_buffer, buffers[1], batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    cout << "infer_done" << endl;
}

// 释放资源
CornerDetect::~CornerDetect()
{
     
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[0]));
    CUDA_CHECK(cudaFree(buffers[1]));


    CUDA_CHECK(cudaFree(gpu_input));
    CUDA_CHECK(cudaFree(gpu_output));
    CUDA_CHECK(cudaFree(gpu_priors));
    CUDA_CHECK(cudaFree(gpu_variances));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();



    delete data;
    delete output_buffer;
    delete decode_bbox;
    delete output;
    
    
}




int main()
{
    string name = "../data/0.jpg"; 
    string bbox_name = "../data/bbox.bin";      
    // CornerDetect pred("../data/CORNER-NEW-MERGE.engine",bbox_name);
    CornerDetect pred("../data/CORNER-TEST.engine",bbox_name);

    for (int i=0;i<1000;++i)
    {
        pred.preprocess(name);  
        pred.infer();
        pred.postprocess(name,0.3,0.5);     
    }
                    
              
    return 0;
}

