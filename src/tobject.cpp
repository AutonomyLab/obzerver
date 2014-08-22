#include "obzerver/tobject.hpp"
#include "glog/logging.h"
TObject::TObject(const std::size_t hist_len, const float fps)
  :
    hist_len(hist_len),
    fps(fps),
    obj_hist(hist_len),
    self_similarity(hist_len, false), // TODO
    periodicity(hist_len, fps)
{
  ;
}

void TObject::Reset() {
  obj_hist.clear();
  self_similarity.Reset();
}

void TObject::Update(const cv::Mat &frame, const object_t &obj, const bool reset) {
  if (reset) Reset();
  obj_hist.push_front(obj);
  self_similarity.Update(frame(obj.bb).clone());
  if (self_similarity.IsFull()) {
    for (int i = 0; i < self_similarity.GetSimMatrix().cols; i+=15) {
      // First time, reset the spectrum, then add up the power
      LOG(INFO) << "Trying with " << i;
      periodicity.Update(self_similarity.GetSimMatrix().row(i), i != 0, false);
    }
    LOG(INFO) << "Avg Spectrum: " << cv::Mat(periodicity.GetSpectrum(), false).t();
  }
}

void TObject::Update(const cv::Mat &frame, const cv::Rect &bb, const bool reset) {
  object_t obj(bb);
  Update(frame, obj, reset);
}
