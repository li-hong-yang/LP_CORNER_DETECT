#ifndef _DECODE_UTILS_H_
#define _DECODE_UTILS_H_



#include <iostream>
#include "opencv2/opencv.hpp"
#include "corner_detect.h"

using namespace std;

cv::Mat get_perspective_mat(float corner[]);
cv::Rect get_rect(float bbox[4]);
float* DecodeBbox(float* loc,float* priors,vector<float>& variances,int w,int h);
float* DecodeCorner(float* loc,float* priors,vector<float>& variances);
float iou(float lbox[4], float rbox[4]);
bool cmp(const Yolo::Detection& a, const Yolo::Detection& b);
#endif
