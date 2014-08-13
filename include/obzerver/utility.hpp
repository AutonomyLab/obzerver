#ifndef _UTILITY_HPP
#define _UTILITY_HPP

#include <vector>
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpumat.hpp"

#include <iostream>
// TODO: Don't change the vec
bool CalcVecDFT(std::vector<float>& vec, std::vector<float>& fft_power, const std::vector<float>& win, const unsigned int remove_count = 0, const bool verbose = false);

/* Misc */

template<class T>
inline cv::Point_<T> rectCenter(const cv::Rect_<T>& r) {
    return cv::Point_<T>(r.x + (r.width/static_cast<T>(2)), r.y + (r.height/static_cast<T>(2)));
}

template<class T>
inline cv::Rect_<T> ClampRect(const cv::Rect_<T>& rect, const cv::Rect_<T>& boundery) {
  cv::Rect_<T> res = rect;
  res.x = clamp(rect.x, boundery.tl().x, boundery.br().x);
  res.y = clamp(rect.y, boundery.tl().y, boundery.br().y);
  T x2 = clamp(rect.br().x, boundery.tl().x, boundery.br().x);
  T y2 = clamp(rect.br().y, boundery.tl().y, boundery.br().y);
  res.width = x2 - res.x;
  res.height = y2 - res.y;
  return res;
}

inline cv::Point2f transformPoint(const cv::Point2f& pt, const cv::Mat& m) {
    CV_Assert(m.cols == 3 && m.rows == 3);
    cv::Mat_<double> pt_vec = cv::Mat_<double>(3, 1);
    pt_vec[0][0] = pt.x;
    pt_vec[1][0] = pt.y;
    pt_vec[2][0] = 1.0f;

    cv::Mat_<double> pt_res = m * pt_vec;
    return cv::Point2f(pt_res[0][0] / pt_res[2][0], pt_res[0][1] / pt_res[2][0]);
}

class trackbar_data_t {
public:
  cv::VideoCapture* cap_p;
  unsigned long int* frame_num_p;
  trackbar_data_t(cv::VideoCapture* cap_p, unsigned long int* frame_num_p):
    cap_p(cap_p), frame_num_p(frame_num_p) {;}
};

inline void trackbarCallback(int pos, void* data) {
    //cv::VideoCapture* cap = (cv::VideoCapture*) data;
  trackbar_data_t* t_data = (trackbar_data_t*) data;
  if (pos == (int) *(t_data->frame_num_p)) return;
  t_data->cap_p->set(CV_CAP_PROP_POS_FRAMES, pos);
  *(t_data->frame_num_p) = pos;
}

template <typename T> inline T clamp (T x, T a, T b)
{
    return ((x) > (a) ? ((x) < (b) ? (x) : (b)) : (a));
}

template <typename T> inline T mapValue(T x, T a, T b, T c, T d)
{
    x = clamp(x, a, b);
    return c + (d - c) * (x - a) / (b - a);
}

// TODO: Why const does not work?
template<typename T> inline
T quick_median(std::vector<T> &v)
{
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

// From: https://github.com/Itseez/opencv/blob/master/samples/cpp/fback.cpp
void drawOpticalFlowVectors(const cv::Mat& flow_x, const cv::Mat& flow_y, cv::Mat& cflowmap, int step, const cv::Scalar& color);
void drawFeaturePoints(cv::Mat& frame, const std::vector<cv::Point2f>& points, const cv::Scalar& color, int rad = 2);
void drawWeightedFeaturePoints(
        cv::Mat& frame,
        const std::vector<cv::Point2f>& points,
        const std::vector<cv::Size2f>& sizes,
        const std::vector<cv::Point2f>& displacements,
        const std::vector<double>& weights,
        const cv::Scalar& color);

void drawFeaturePointsTrajectory(cv::Mat& frame,
                                 const cv::Mat& outliers,
                                 const std::vector<cv::Point2f>& points_prev, const std::vector<cv::Point2f>& points_cur,
                                 int rad,
                                 const cv::Scalar& color_prev, const cv::Scalar& color_cur, const cv::Scalar& color_line);

void drawFlowField(const cv::Mat& flow_x, const cv::Mat& flow_y, cv::Mat& flowField,
                   bool auto_max = false, float* max_x = 0, float* max_y = 0);
void drawOpticalFlowScatter(const cv::Mat& u,
                            const cv::Mat& v,
                            cv::Mat& frame,
                            const float max_val,
                            const unsigned int frame_size,
                            const unsigned int offset_x = 0,
                            const unsigned int offset_y = 0);

std::string getCvMatTypeStr(const int type);

void dumpCvMatInfo(const cv::Mat& m);

void DownloadGpuMatToVecFC2(const cv::gpu::GpuMat& d_mat, std::vector<cv::Point2f>& vec);
void DownloadGpuMatToVecUC1(const cv::gpu::GpuMat& d_mat, std::vector<uchar>& vec);

#endif
