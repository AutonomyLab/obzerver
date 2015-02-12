#ifndef TOBJECT_HPP
#define TOBJECT_HPP

#include "obzerver/common_types.hpp"
#include "obzerver/circular_buffer.hpp"
#include "obzerver/self_similarity.hpp"
#include "obzerver/fft.hpp"

namespace obz
{

class TObject {
protected:
  std::size_t hist_len;
  float fps;
  CircularBuffer<object_t> obj_hist;
  SelfSimilarity self_similarity;
  Periodicity periodicity;

public:
  TObject(const std::size_t hist_len, const float fps);
  void Reset();
  void Update(const object_t& obj, const cv::Mat& frame, const bool reset = false);
  void Update(const cv::Rect& bb, const cv::Mat& frame, const bool reset = false);

  const object_t& Get(const std::size_t index = 0) const {return obj_hist.at(index); }
  const CircularBuffer<object_t>& GetHist() const {return obj_hist;}
  const CircularBuffer<object_t>& operator()() const {return obj_hist;}

  const Periodicity& GetPeriodicity() const {return periodicity; }
  const SelfSimilarity& GetSelfSimilarity() const {return self_similarity; }
};

}  // namespace obz
#endif
