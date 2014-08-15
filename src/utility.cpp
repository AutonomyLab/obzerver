#include "obzerver/utility.hpp"
#include <iostream>
#include <cassert>

std::string getCvMatTypeStr(const int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void dumpCvMatInfo(const cv::Mat &m) {
    std::cout << m.size() << " : " << m.dims << " : " << getCvMatTypeStr(m.type()) << std::endl;
}

void drawOpticalFlowVectors(const cv::Mat& flow_x, const cv::Mat& flow_y, cv::Mat& cflowmap, int step, const cv::Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const cv::Point2f& fxy = cv::Point2f(flow_x.at<float>(y,x), flow_y.at<float>(y,x));
//            if (sqrt(fxy.x * fxy.x + fxy.y * fxy.y) > 1.0) {
            //if ((fabs(fxy.x) > 1.0) || (fabs(fxy.y) > 1.0)) {
//                cv::line(cflowmap,
//                         cv::Point(x,y),
//                         cv::Point(x+fxy.x, y+fxy.y),
//                         color);

            if (fabsf(fxy.x) > 1.0)
                cv::line(cflowmap,
                     cv::Point(x,y),
                     cv::Point(x+fxy.x, y),
                     cv::Scalar(0, 0 , 255));

            if (fabsf(fxy.y) > 1.0)
                cv::line(cflowmap,
                     cv::Point(x,y),
                     cv::Point(x,y+fxy.y),
                     cv::Scalar(0, 255 , 0));
//            }
            //cv::circle(cflowmap, cv::Point(x,y), 1, color);
        }
}

void DownloadGpuMatToVecFC2(const cv::gpu::GpuMat& d_mat, std::vector<cv::Point2f>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

void DownloadGpuMatToVecUC1(const cv::gpu::GpuMat& d_mat, std::vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}


void drawFeaturePoints(cv::Mat& frame, const std::vector<cv::Point2f>& points, const cv::Scalar& color, int rad) {
    for (unsigned int i = 0; i < points.size(); i++) {
        cv::circle(frame, points.at(i), rad, color, -1);
    }
}

void drawWeightedFeaturePoints(
        cv::Mat& frame,
        const std::vector<cv::Point2f>& points,
        const std::vector<cv::Size2f>& sizes,
        const std::vector<cv::Point2f>& displacements,
        const std::vector<double>& weights,
        const cv::Scalar& color) {

    const double base_weight = points.size();
    for (unsigned int i = 0; i < points.size(); i++) {
        //std::cout << weights.at(i) << std::endl;
        cv::circle(frame, points.at(i), (int) round(weights.at(i) * base_weight), color, -1);
        cv::Rect bb;
        bb.x = points[i].x - sizes[i].width/2.0;
        bb.y = points[i].y - sizes[i].height/2.0;
        bb.width = sizes[i].width;
        bb.height = sizes[i].height;
        //bb.br = points.at(i) + cv::Point2f(sizes[i].width/2.0, sizes[i].height/2.0);
        //cv::rectangle(frame, bb, color);
        //cv::line(frame, points.at(i), points.at(i) - displacements.at(i), color);
    }
}

void drawFeaturePointsTrajectory(cv::Mat& frame,
                                 const cv::Mat& outliers,
                                 const std::vector<cv::Point2f>& points_prev, const std::vector<cv::Point2f>& points_cur,
                                 int rad,
                                 const cv::Scalar& color_prev, const cv::Scalar& color_cur, const cv::Scalar& color_line)
{
    CV_Assert(points_prev.size() == points_cur.size());
    for (unsigned int i = 0; i < points_prev.size(); i++) {
        if (outliers.data && 1 == outliers.at<uchar>(i, 0)) { // Inlier
            cv::circle(frame, points_prev.at(i), rad, CV_RGB(127, 127, 127), -1);
            cv::circle(frame, points_cur.at(i), rad, CV_RGB(127, 127, 127), -1);
            cv::line(frame, points_prev.at(i), points_cur.at(i), CV_RGB(127, 127, 127));
        } else { // Outlier
            cv::circle(frame, points_prev.at(i), rad, color_prev, -1);
            cv::circle(frame, points_cur.at(i), rad, color_cur, -1);
            cv::line(frame, points_prev.at(i), points_cur.at(i), color_line);
        }
    }
}


void drawFlowField(const cv::Mat& flow_x, const cv::Mat& flow_y, cv::Mat& flowField,
                   bool auto_max, float* max_x, float* max_y)
{
    if (flowField.channels() == 3)
        cv::cvtColor(flowField, flowField, cv::COLOR_BGR2GRAY);
    cv::Mat cache_grayscale(flowField);


    float maxDisplacement_x = 0.0f;
    float maxDisplacement_y = 0.0f;

    if (auto_max) {
        for (int i = 0; i < flow_x.rows; ++i)
        {
            const float* ptr_x = flow_x.ptr<float>(i);
            const float* ptr_y = flow_y.ptr<float>(i);

            for (int j = 0; j < flow_x.cols; ++j)
            {
                //float d = cv::max(fabsf(ptr_u[j]), fabsf(ptr_v[j]));

                maxDisplacement_x = cv::max(fabsf(ptr_x[j]), maxDisplacement_x);
                maxDisplacement_y = cv::max(fabsf(ptr_y[j]), maxDisplacement_y);
            }
        }
        std::cout << "Setting Max To " << maxDisplacement_x << " " << maxDisplacement_y << std::endl;
        if (max_x && max_y) {
            *max_x = maxDisplacement_x;
            *max_y = maxDisplacement_y;
            //
        }
    } else {
        maxDisplacement_x = *max_x;
        maxDisplacement_y = *max_y;
        //std::cout << "Reading Max To " << maxDisplacement_x << " " << maxDisplacement_y << std::endl;
    }


    flowField = cv::Mat::zeros(cv::Size(flow_x.size().width * 2, flow_x.size().height), CV_8UC4);
    //flowField.create(cv::Size(u.size().width * 2, u.size().height), CV_8UC4);

    for (int i = 0; i < flowField.rows; ++i)
    {
        const float* ptr_x = flow_x.ptr<float>(i);
        const float* ptr_y = flow_y.ptr<float>(i);

        //const uchar* ptr_cache = cache_grayscale.ptr<uchar>(i);
        cv::Vec4b* row = flowField.ptr<cv::Vec4b>(i);

        const int half_width = flowField.cols / 2.0;
        for (int j = 0; j < half_width; ++j)
        {
            //row[j][0] = ptr_cache[j];
            //row[j+half_width][0] = ptr_cache[j];

            //if (sqrt(ptr_x[j] * ptr_x[j] + ptr_y[j] * ptr_y[j]) > 1.0) {
            if (fabsf(ptr_x[j]) > 1.0) // x is red
                //row[j][2] = static_cast<unsigned char> (mapValue<float> (ptr_x[j], -maxDisplacement_x, maxDisplacement_x, 0.0f, 255.0f));
                row[j][2] = static_cast<unsigned char> (mapValue<float> (fabsf(ptr_x[j]), 0.0, maxDisplacement_x, 0.0f, 255.0f));

            if (fabsf(ptr_y[j]) > 1.0) // y is green
//                row[j+half_width][1] = static_cast<unsigned char> (mapValue<float> ( ptr_y[j], -maxDisplacement_y, maxDisplacement_y, 0.0f, 255.0f));
                row[j+half_width][1] = static_cast<unsigned char> (mapValue<float> ( fabsf(ptr_y[j]), 0.0, maxDisplacement_y, 0.0f, 255.0f));
//                if (!auto_max)
//                    std::cout << "Final Values: " << (int) row[j][1] << " " << (int) row[j+half_width][2] << std::endl;
            /* else {
                row[j][1] = 0;
                row[j+half_width][2] = 0;
            }*/
            row[j][3] = 255;
        }
    }
}

void drawOpticalFlowScatter(const cv::Mat& u,
                            const cv::Mat& v,
                            cv::Mat& frame,
                            const float max_val,
                            const unsigned int frame_size,
                            const unsigned int offset_x,
                            const unsigned int offset_y) {

    CV_Assert(u.size() == v.size());
    frame = cv::Mat::zeros(cv::Size(frame_size, frame_size), CV_8UC3);
    cv::line(frame, cv::Point(0, frame_size/2), cv::Point(frame_size, frame_size/2), cv::Scalar(255, 255, 255));
    cv::line(frame, cv::Point(frame_size/2, 0), cv::Point(frame_size/2, frame_size), cv::Scalar(255, 255, 255));
    for (unsigned int i = offset_y; i < u.rows-offset_y; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);
        for (unsigned int j = offset_x; j < u.cols-offset_x; ++j)
        {
            if (sqrt(ptr_u[j] * ptr_u[j] + ptr_v[j] * ptr_v[j]) <= 1.0) continue;
            cv::circle(frame,
                       cv::Point(
                           static_cast<int> (mapValue(ptr_u[j], -max_val, max_val, 0.0f, (float) frame_size)),
                           static_cast<int> (mapValue(ptr_v[j], -max_val, max_val, 0.0f, (float) frame_size))
                       ),
                       1,
                       cv::Scalar(0, 255, 0)
           );
        }

    }

}
