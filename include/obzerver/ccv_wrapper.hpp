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
 * Only grayscale and 3-channel RGB images are accepted.
 * if roi.size() > 0, performs roi extraction
 *
 * Very Important Note: CCV Assumes RGB byte order for 24bit images, OpenCV color images
 * are BGR by default. It is the user's responsibility to make sure that input opencv_mat color
 * images are converted to RGB in case before passing them to this function.
 *
 */
ccv_dense_matrix_t* FromOpenCV(const cv::Mat& opencv_mat, const cv::Rect& roi = cv::Rect(0, 0, 0, 0));

// Base class to enable/disable ccv_cache
class CommonBase
{
private:
  static bool inited_;
public:
  CommonBase();
  ~CommonBase();
};

class ICFCascadeClassifier: public CommonBase
{
protected:
  std::string cascade_file_;
  ccv_icf_classifier_cascade_t* cascade_ptr_;
  ccv_icf_param_t cascade_params_;
public:
  struct result_t
  {
    cv::Rect bb;
    float confidence;
    std::size_t neighbors;
    result_t(const cv::Rect& bb_, const float confidence_, const std::size_t& neighbors_)
      : bb(bb_), confidence(confidence_), neighbors(neighbors_) {;}
  };

  // Default values from ccv_icf.c
  ICFCascadeClassifier(
      const std::string& cascade_file,
      const std::int32_t min_neighbors = 2,
      const std::int32_t step_through = 2,
      const std::int32_t interval = 8,
      const float threshold = 0.0);

  ~ICFCascadeClassifier();
  void SetParams(
      const std::int32_t min_neighbors,
      const std::int32_t step_through,
      const std::int32_t interval,
      const float threshold);

  std::size_t Detect(
      const cv::Mat& frame,
      std::vector<result_t>& result,
      const cv::Rect& roi = cv::Rect(0, 0, 0, 0));
};

}  // namespace ccv

#endif  // CCV_WRAPPER_HPP
