#include "obzerver/tobject.hpp"
#include "glog/logging.h"

namespace obz
{

TObject::TObject(const std::size_t hist_len, const float fps)
  :
    hist_len(hist_len),
    fps(fps),
    obj_hist(hist_len),
    motion_hist(hist_len),
    optflow_hist(hist_len),
    sequence(hist_len)
//    self_similarity(hist_len, false), // TODO
//    periodicity(hist_len, fps)
{
  ;
}

void TObject::Reset() {
  obj_hist.clear();
//  self_similarity.Reset();
}

void TObject::Update(const object_t &obj,
                     const cv::Mat &frame,
                     const cv::Mat &diff_image,
                     const float flow,
                     const bool reset) {
  if (reset) Reset();
//  LOG(INFO) << "[TObj] Updating with: " << obj.bb << " | Reset: " << reset;
  obj_hist.push_front(obj);
  sequence.push_front(frame(obj.bb).clone());

  const float num_pixels = static_cast<float>(obj.bb.area());
  const float num_nonzero_pixels = static_cast<float>(cv::countNonZero(diff_image(obj.bb))) + 1.0;
  LOG(INFO) << "[OBJ] BB num_pixels: " << num_pixels << " non-zero: " << num_nonzero_pixels;

  motion_hist.push_front(cv::sum(diff_image(obj.bb))[0] / num_nonzero_pixels);
  optflow_hist.push_front(flow);

  // Only calculate the self similarity matrix when:
  // 1) The caller explicitly asked us to
//  self_similarity.Update(frame(obj.bb).clone());
//  if (self_similarity.IsFull()) {
//    for (int i = 0; i < self_similarity.GetSimMatrix().cols; i+=1) {
//      // First time, reset the spectrum, then add up the power
//      periodicity.Update(self_similarity.GetSimMatrix().row(i), i != 0, false);
//    }
//    LOG(INFO) << "Avg Spectrum: " << cv::Mat(periodicity.GetSpectrum(), false).t();
//  }
}

void TObject::Update(const cv::Rect &bb,
                     const cv::Mat &frame,
                     const cv::Mat &diff_image, const float flow,
//                     const pts_vec_t &prev_pts,
//                     const pts_vec_t &curr_pts,
                     const bool reset) {
  object_t obj(bb);
  Update(obj, frame, diff_image, flow, reset);
}

}  // namespace obz
