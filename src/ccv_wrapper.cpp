#include "obzerver/ccv_wrapper.hpp"
#include <exception>

namespace ccv
{

ccv_dense_matrix_t* FromOpenCV(const cv::Mat& opencv_mat, const cv::Rect& roi)
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

// CCV Base Class

bool CommonBase::inited_ = false;

CommonBase::CommonBase()
{
  if (!inited_) ccv_enable_default_cache();
}

CommonBase::~CommonBase()
{
  if (inited_) ccv_disable_cache();
}

// CCV ICF

ICFCascadeClassifier::ICFCascadeClassifier(
    const std::string &cascade_file,
    const std::int32_t min_neighbors,
    const std::int32_t step_through,
    const std::int32_t interval,
    const float threshold)
  : CommonBase(),
    cascade_file_(cascade_file),
    cascade_ptr_(ccv_icf_read_classifier_cascade(cascade_file_.c_str()))
{
  if (!cascade_ptr_)
  {
    throw std::runtime_error("ccv:: Error loading cascade file: " + cascade_file_);
  }
  SetParams(min_neighbors, step_through, interval, threshold);
}

ICFCascadeClassifier::~ICFCascadeClassifier()
{
  if (cascade_ptr_) ccv_icf_classifier_cascade_free(cascade_ptr_);
}

void ICFCascadeClassifier::SetParams(
    const std::int32_t min_neighbors,
    const std::int32_t step_through,
    const std::int32_t interval,
    const float threshold)
{
  cascade_params_.min_neighbors = min_neighbors;
  cascade_params_.step_through = step_through;
  cascade_params_.interval = interval;
  cascade_params_.threshold = threshold;
  cascade_params_.flags = 0;
}

std::size_t ICFCascadeClassifier::Detect(const cv::Mat &frame, std::vector<result_t> &result, const cv::Rect &roi)
{
  ccv_dense_matrix_t* ccv_frame = ccv::FromOpenCV(frame, roi);
  ccv_array_t* objects = ccv_icf_detect_objects(ccv_frame, &cascade_ptr_, 1, cascade_params_);
  const std::size_t num_objects = static_cast<std::size_t>(objects->rnum);

  result.reserve(num_objects);
  for (std::int32_t i = 0; i < objects->rnum; i++)
  {
    ccv_comp_t* obj = reinterpret_cast<ccv_comp_t*>(ccv_array_get(objects, i));
    result.push_back(result_t(cv::Rect(obj->rect.x, obj->rect.y, obj->rect.width, obj->rect.height),
                                obj->classification.confidence,
                                obj->neighbors));
  }
  ccv_array_free(objects);
  return num_objects;
}
}  // namespace ccv
