#ifndef TOBJECT_HPP
#define TOBJECT_HPP

#include "obzerver/circular_buffer.hpp"
#include "obzerver/self_similarity.hpp"
#include "obzerver/fft.hpp"

struct object_t {
  cv::Rect bb;
  object_t(): bb(cv::Rect(0,0,0,0)) {;}
  object_t(const cv::Rect& bb): bb(bb) {;}
};

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
  void Update(const cv::Mat& frame, const object_t& obj, const bool reset = false);
  void Update(const cv::Mat& frame, const cv::Rect& bb, const bool reset = false);

  const object_t& Get(const std::size_t index = 0) const {return obj_hist.at(index); }
  const CircularBuffer<object_t>& GetHist() const {return obj_hist;}
  const CircularBuffer<object_t>& operator()() const {return obj_hist;}

  const Periodicity& GetPeriodicity() const {return periodicity; }
  const SelfSimilarity& GetSelfSimilarity() const {return self_similarity; }
};

#endif
