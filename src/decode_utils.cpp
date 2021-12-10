#include "decode_utils.h"


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

cv::Rect get_rect(float bbox[4],float fx,float fy) {
    int x = int(bbox[0]*fx);
    int y = int(bbox[1]*fy);
    int w = int(bbox[2]*fx-bbox[0]*fx);
    int h = int(bbox[3]*fy-bbox[1]*fy); 
    return cv::Rect(x,y,w,h);
}


void DecodeCorner(float* loc,float* priors,vector<float>& variances,float* bbox)
{


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


void DecodeBbox(float* loc,float* priors,vector<float>& variances,float* dst)
{
    float bbox[4];

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
    
}