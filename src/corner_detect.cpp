#include <iostream>
#include <chrono>
#include "cuda_utils.h"
#include "logging.h"
#include "corner_detect.h"
#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>

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

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.conf > b.conf;
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}


cv::Rect get_rect(float bbox[4]) {
    int x = int(bbox[0]);
    int y = int(bbox[1]);
    int w = int(bbox[2]-bbox[0]);
    int h = int(bbox[3]-bbox[1]);

    
    return cv::Rect(x,y,w,h);
}


float* DecodeBbox(float* loc,float* priors,vector<float>& variances,int w,int h)
{



    float* bbox = new float[4];
    float* dst = new float[4];

    bbox[0] = priors[0] + loc[0] * variances[0] * priors[2];
    bbox[1] = priors[1] + loc[1] * variances[0] * priors[3];

    bbox[2] = priors[2] * exp(loc[2]*variances[1]);
    bbox[3] = priors[3] * exp(loc[3]*variances[1]);

    dst[0] = (bbox[0] - bbox[2]/2.0);
    dst[1] = (bbox[1] - bbox[3]/2.0);
    dst[2] = (bbox[0] + bbox[2]/2.0);
    dst[3] = (bbox[1] + bbox[3]/2.0);

    dst[0] *= 416;
    dst[1] *= 416;
    dst[2] *= 416;
    dst[3] *= 416;
    return dst;
    
}


float* DecodeCorner(float* loc,float* priors,vector<float>& variances)
{



    float* bbox = new float[8];

    bbox[0] = priors[0] + loc[0] * variances[0] * priors[2];
    bbox[1] = priors[1] + loc[1] * variances[0] * priors[3];

    bbox[2] = priors[0] + loc[2] * variances[0] * priors[2];
    bbox[3] = priors[1] + loc[3] * variances[0] * priors[3];

    bbox[4] = priors[0] + loc[4] * variances[0] * priors[2];
    bbox[5] = priors[1] + loc[5] * variances[0] * priors[3];

    bbox[6] = priors[0] + loc[6] * variances[0] * priors[2];
    bbox[7] = priors[1] + loc[7] * variances[0] * priors[3];

   
    bbox[0] *= 416;
    bbox[1] *= 416;
    bbox[2] *= 416;
    bbox[3] *= 416;
    bbox[4] *= 416;
    bbox[5] *= 416;
    bbox[6] *= 416;
    bbox[7] *= 416;
    return bbox;
    
}



cv::Mat get_perspective_mat(float corner[])
{
	cv::Point2f src_points[] = { 
		cv::Point2f(int(corner[4]), int(corner[5])),
		cv::Point2f(int(corner[6]), int(corner[7])),
		cv::Point2f(int(corner[2]), int(corner[3])),
		cv::Point2f(int(corner[0]), int(corner[1])) };
 
	cv::Point2f dst_points[] = {
		cv::Point2f(0, 0),
		cv::Point2f(94,0),
		cv::Point2f(0, 24),
		cv::Point2f(94, 24) };
 
	cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);
	
	return M;
 
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

    for (int i = 0;i < output_size/det_size; i++) {

        // if (i==0) 
        // {
        //     cout << output_buffer[i] << endl;
        //     cout << output_buffer[i+1] << endl;
        //     cout << output_buffer[i+2] << endl;
        //     cout << output_buffer[i+3] << endl;
        //     cout << output_buffer[i+4] << endl;
        //     cout << output_buffer[i+5] << endl;
        //     cout << output_buffer[i+6] << endl;

        // }
        
        if (output_buffer[det_size * i + 5] <= conf_thresh) continue;
        Yolo::Detection det;
        float loc[4];
        float priors[4];
        float pre[8];
        float* xyxy;
        float* cooners;
        memcpy(&loc, &output_buffer[det_size * i], 4*sizeof(float));
        memcpy(&priors, (char*)decode_bbox+sizeof(float)*4*i, 4*sizeof(float));
        memcpy(&pre, &output_buffer[det_size*i+6], 8*sizeof(float));
        
        xyxy = DecodeBbox(loc,priors,variances,imgw,imgh);
        cooners = DecodeCorner(pre,priors,variances);
    
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


            cout << cooners[0] << endl;
            cout << cooners[1] << endl;
            cout << cooners[2] << endl;
            cout << cooners[3] << endl;
            cout << cooners[4] << endl;
            cout << cooners[5] << endl;
            cout << cooners[6] << endl;
            cout << cooners[7] << endl;
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
        cv::Rect r = get_rect(res[j].bbox);
        cv::rectangle(imgraw, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(imgraw, "lp", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
    cv::imwrite("lp.jpg", imgraw);

    cout << res[0].corner[0] << endl;
    cout << res[0].corner[1] << endl;
    cout << res[0].corner[2] << endl;
    cout << res[0].corner[3] << endl;
    cout << res[0].corner[4] << endl;
    cout << res[0].corner[5] << endl;
    cout << res[0].corner[6] << endl;
    cout << res[0].corner[7] << endl;

    cv::Mat pers_img;
    cv::Mat out_img;
    pers_img = get_perspective_mat(res[0].corner);
    cv::warpPerspective(imgraw,out_img,pers_img,cv::Size(94,24));

    cv::imwrite("pers_lp.jpg", out_img);





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

