#ifndef _UTILITY_HPP
#define _UTILITY_HPP

#include <vector>
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpumat.hpp"

class StepBenchmarker {
private:
    long int last_tick;
    typedef std::pair<std::string, double> tick_t;
    std::vector<tick_t> items;
    double update();

public:
    StepBenchmarker();
    void reset();
    void tick();
    void tick(const std::string& text);
    void dump(const bool clear_screen = false) const;
};

// TODO: Don't change the vec
bool CalcVecDFT(std::vector<float>& vec, std::vector<float>& fft_power, const std::vector<float>& win, const unsigned int remove_count = 0, const bool verbose = false);

/* Misc */

inline cv::Point2f rectCenter(const cv::Rect& r) {
    return cv::Point2f(r.x + (r.width/2), r.y + (r.height/2));
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

inline void trackbarCallback(int pos, void* data) {
    cv::VideoCapture* cap = (cv::VideoCapture*) data;
    cap->set(CV_CAP_PROP_POS_FRAMES, pos);
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
