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

static Logger gLogger;

using namespace nvinfer1;
using namespace std;

// 加载模型，分配显存和内存
CornerDetect::CornerDetect(const std::string & engine_name)
{

    cudaSetDevice(device);

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
    
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);

    int inputIndex, outputIndex;
    for (int bi = 0; bi < engine->getNbBindings(); bi++)
    {
        if (engine->bindingIsInput(bi) == true)
        {
            inputIndex = bi;
            // printf("Binding %d (%s): Input.\n", bi, engine->getBindingName(bi));
        }
        else
        {
            outputIndex = bi;
            // printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
        }
    }

    // const int inputIndex = engine->getBindingIndex("input");
    // const int outputIndex = engine->getBindingIndex("output");
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


void CornerDetect::postprocess(string& img_name,float conf_thresh,float nms_thresh,string& bbox)
{

    float* decode_bbox = new float[7098*4];


    assert(decode_bbox != nullptr);
    vector<float> variances = {0.1,0.2};

    std::ifstream in(bbox, std::ios::in | std::ios::binary); 
    in.read((char *)decode_bbox, sizeof(float)*7098*4);


    std::vector<Yolo::Detection> res;
    int det_size = 14;
    std::vector<Yolo::Detection> dst;
    cv::Mat img = cv::imread(img_name);
    cv::Mat imgraw;
    cv::resize(img,imgraw,cv::Size(416,416));
    int imgh = img.rows;
    int imgw = img.cols;

    float fx = imgw/416.0;
    float fy = imgh/416.0;


    float* xyxy = new float[4];
    float* cooners = new float[8];

    for (int i = 0;i < output_size/det_size; i++) {      
        if (output_buffer[det_size * i + 5] <= conf_thresh) continue;
        Yolo::Detection det;
        float loc[4];
        float priors[4];
        float pre[8];
       

        // vector<float> m1(output_buffer,output_buffer+4);
        // cout << m1[0] << endl;

        memcpy(loc, &output_buffer[det_size * i], 4*sizeof(float));
        memcpy(priors, (char*)decode_bbox+sizeof(float)*4*i, 4*sizeof(float));
        memcpy(pre, &output_buffer[det_size*i+6], 8*sizeof(float));
        
        DecodeBbox(loc,priors,variances,xyxy);
        DecodeCorner(pre,priors,variances,cooners);
    
        // if (xyxy[0]<0 || xyxy[1]<0 || xyxy[2]<0 || xyxy[3] < 0) continue;
        // cout << xyxy[0] << endl;
        det.conf = output_buffer[det_size * i+5];
        memcpy(&det.corner, &output_buffer[det_size * i+6], 8*sizeof(float));
        memcpy(&det.bbox, xyxy, 4*sizeof(float));
        memcpy(&det.corner, cooners, 8*sizeof(float));
        dst.push_back(det);

        if (i==0) 
        {
            // cout <<det.conf << endl;
            // cout <<det.bbox[0] << endl;
            // cout <<det.bbox[1] << endl;
            // cout <<det.bbox[2] << endl;
            // cout <<det.bbox[3] << endl;


            // cout <<det.corner[0] << endl;
            // cout <<det.corner[1] << endl;


            // cout << pre[0] << endl;
            // cout << pre[1] << endl;
            // cout << pre[2] << endl;
            // cout << pre[3] << endl;
            // cout << pre[4] << endl;
            // cout << pre[5] << endl;
            // cout << pre[6] << endl;
            // cout << pre[7] << endl;


            // cout << cooners[0] << endl;
            // cout << cooners[1] << endl;
            // cout << cooners[2] << endl;
            // cout << cooners[3] << endl;
            // cout << cooners[4] << endl;
            // cout << cooners[5] << endl;
            // cout << cooners[6] << endl;
            // cout << cooners[7] << endl;

            // cout << loc[i+3] << endl;

            // cout << priors[i+0] << endl;
            // cout << priors[i+1] << endl;
            // cout << priors[i+2] << endl;
            // cout << priors[i+3] << endl;

            // cout << xyxy[0] << endl;
            // cout << xyxy[i+1] << endl;
            // cout << xyxy[i+2] << endl;
            // cout << xyxy[i+3] << endl;
            // cout << output_buffer[det_size * i+5] << endl;

        }
    }

    delete xyxy;
    delete cooners;
   
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

    // cout << res.size() << endl;
    // cout << res[0].bbox[0] << endl;
    // cout << res[0].bbox[1] << endl;
    // cout << res[0].bbox[2] << endl;
    // cout << res[0].bbox[3] << endl;


    
    for (size_t j = 0; j < 1; j++) {
        cv::Rect r = get_rect(res[j].bbox,fx,fy);
        cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(img, "lp", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
    cv::imwrite("lp.jpg", img);

    // cout << res[0].corner[0] << endl;
    // cout << res[0].corner[1] << endl;
    // cout << res[0].corner[2] << endl;
    // cout << res[0].corner[3] << endl;
    // cout << res[0].corner[4] << endl;
    // cout << res[0].corner[5] << endl;
    // cout << res[0].corner[6] << endl;
    // cout << res[0].corner[7] << endl;

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
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();



    delete [] data;
    delete [] output_buffer;
    
    
}




int main()
{
    CornerDetect pred("../data/CORNER-NEW-MERGE.engine");
    string name = "../data/0.jpg"; 
    string bbox_name = "../data/bbox.bin";                      
    pred.preprocess(name);  
    pred.infer();
    pred.postprocess(name,0.3,0.5,bbox_name);                
    return 0;
}

