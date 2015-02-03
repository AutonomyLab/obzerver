#include "obzerver/ccv_wrapper.hpp"
#include <exception>

namespace ccv
{

// CCV Base Class

bool CCVBase::inited_ = false;

CCVBase::CCVBase()
{
  if (!inited_) ccv_enable_default_cache();
}

CCVBase::~CCVBase()
{
  if (inited_) ccv_disable_cache();
}

// CCV ICF

CCV_ICF::CCV_ICF(
    const std::string &cascade_file,
    const std::int32_t min_neighbors,
    const std::int32_t step_through,
    const std::int32_t interval,
    const float threshold)
  : CCVBase(),
    cascade_file_(cascade_file),
    cascade_ptr_(ccv_icf_read_classifier_cascade(cascade_file_.c_str()))
{
  if (!cascade_ptr_)
  {
    throw std::runtime_error("ccv:: Error loading cascade file: " + cascade_file_);
  }
  SetParams(min_neighbors, step_through, interval, threshold);
}

CCV_ICF::~CCV_ICF()
{
  if (cascade_ptr_) ccv_icf_classifier_cascade_free(cascade_ptr_);
}

void CCV_ICF::SetParams(
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

std::size_t CCV_ICF::Detect(const cv::Mat &frame, std::vector<CCV_Result> &result, const cv::Rect &roi)
{
  ccv_dense_matrix_t* ccv_frame = ccv::FromOpenCV(frame, roi);
  ccv_array_t* objects = ccv_icf_detect_objects(ccv_frame, &cascade_ptr_, 1, cascade_params_);
  const std::size_t num_objects = static_cast<std::size_t>(objects->rnum);

  result.reserve(num_objects);
  for (std::int32_t i = 0; i < objects->rnum; i++)
  {
    ccv_comp_t* obj = reinterpret_cast<ccv_comp_t*>(ccv_array_get(objects, i));
    result.push_back(CCV_Result(cv::Rect(obj->rect.x, obj->rect.y, obj->rect.width, obj->rect.height),
                                obj->classification.confidence,
                                obj->neighbors));
  }
  ccv_array_free(objects);
  return num_objects;
}
}  // namespace ccv
