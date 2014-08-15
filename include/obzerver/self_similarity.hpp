#ifndef SELF_SIMILARITY_H
#define SELF_SIMILARITY_H

#include "opencv2/core/core.hpp"

#include "obzerver/circular_buffer.hpp"
#include "obzerver/benchmarker.hpp"

class SelfSimilarity {

  typedef CircularBuffer<cv::Mat> mseq_t;

private:
  std::size_t hist_len;
  bool debug_mode;
  mseq_t sequence;
  cv::Mat sim_matrix;

  // For median
  std::vector<std::size_t> widths;
  std::vector<std::size_t> heights;

  cv::Mat render;

  StepBenchmarker& ticker;

  float CalcFramesSimilarity(const cv::Mat &m1, const cv::Mat &m2, cv::Mat &buff) const;
public:
  SelfSimilarity(const std::size_t hist_len, const bool debug_mode = false);
  void Update();
  void Update(const cv::Mat &m, const bool reset = false);
  const cv::Mat& GetSimMatrix() const;
  const cv::Mat GetSimMatrixRendered();
  void WriteToDisk(const std::string &path, const std::string &prefix = std::string("seq")) const;
  bool IsEmpty() const;
  bool IsFull() const;
  void Reset();
  std::size_t GetHistoryLen() const;
};
#endif
