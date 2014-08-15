#ifndef FFT_HPP
#define FFT_HPP

#include "opencv2/core/core.hpp"

bool CalcVecDFT(const cv::Mat& vec_m,
                std::vector<float>& fft_power,
                const cv::Mat& win_m,
                const unsigned int remove_count = 0,
                const bool verbose = false);

class Periodicity {
protected:
  std::size_t hist_len;
  float fps;
  std::vector<float> freqs;
  std::vector<float> fft_power;
  cv::Mat hann_window;

public:
  Periodicity(const std::size_t hist_len, const float fps);
  void Update(const cv::Mat& vec);
  float GetDominantFrequency(const std::size_t start_index = 0) const;
  bool IsPeriodic() const ;
  const std::vector<float>& GetSpectrum() const {return fft_power;}
};

#endif
