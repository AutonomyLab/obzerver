#ifndef CCV_WRAPPER_HPP
#define CCV_WRAPPER_HPP

extern "C"
{
  #include "ccv.h"
}

#include <cstdint>
#include <string>
#include "opencv2/core/core.hpp"

namespace ccv
{

/*
 * Read-only access
 * No copy is performed
 * if roi.size() > 0, performs roi extraction
 */
static ccv_dense_matrix_t* FromOpenCV(const cv::Mat& opencv_mat, const cv::Rect& roi = cv::Rect(0, 0, 0, 0))
{
  CV_Assert(opencv_mat.channels() == 1 || opencv_mat.channels() == 3);
  CV_Assert(opencv_mat.data);

  ccv_dense_matrix_t* ccv_mat = 0;
  if (roi.area() == 0)
  {
    ccv_read(reinterpret_cast<const void*>(opencv_mat.data),
             &ccv_mat,
             (opencv_mat.channels() == 1 ? CCV_IO_GRAY_RAW : CCV_IO_RGB_RAW) | CCV_IO_NO_COPY,
             opencv_mat.rows,
             opencv_mat.cols,
             opencv_mat.cols * opencv_mat.elemSize());
  }
  else
  {
    // ptr(row, col)
    CV_Assert(roi.x >= 0 && roi.y >= 0 && roi.x < opencv_mat.cols && roi.y < opencv_mat.rows);
    ccv_read(reinterpret_cast<const void*>(opencv_mat.ptr(roi.y, roi.x)),
             &ccv_mat,
             (opencv_mat.channels() == 1 ? CCV_IO_GRAY_RAW : CCV_IO_RGB_RAW) | CCV_IO_NO_COPY,
             roi.height,
             roi.width,
             opencv_mat.cols * opencv_mat.elemSize());
  }
  return ccv_mat;
}

// Base class to enable/disable ccv_cache
class CCVBase
{
private:
  static bool inited_;
public:
  CCVBase();
  ~CCVBase();
};

class CCV_ICF: public CCVBase
{
protected:
  std::string cascade_file_;
  ccv_icf_classifier_cascade_t* cascade_ptr_;
  ccv_icf_param_t cascade_params_;
public:
  struct CCV_Result
  {
    cv::Rect bb;
    float confidence;
    std::size_t neighbors;
    CCV_Result(const cv::Rect& bb_, const float confidence_, const std::size_t& neighbors_)
      : bb(bb_), confidence(confidence_), neighbors(neighbors_) {;}
  };

  // Default values from ccv_icf.c
  CCV_ICF(
      const std::string& cascade_file,
      const std::int32_t min_neighbors = 2,
      const std::int32_t step_through = 2,
      const std::int32_t interval = 8,
      const float threshold = 0.0);

  ~CCV_ICF();
  void SetParams(
      const std::int32_t min_neighbors,
      const std::int32_t step_through,
      const std::int32_t interval,
      const float threshold);

  std::size_t Detect(const cv::Mat& frame,
                     std::vector<CCV_Result>& result,
                     const cv::Rect& roi = cv::Rect(0, 0, 0, 0));
};

}  // namespace ccv

#endif  // CCV_WRAPPER_HPP
