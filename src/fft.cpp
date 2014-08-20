#include "glog/logging.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "obzerver/fft.hpp"

#include "obzerver/utility.hpp"

Periodicity::Periodicity(const std::size_t hist_len, const float fps)
  : hist_len(hist_len), fps(fps), hann_window(1, hist_len, CV_32F), average_counter(0)
{
  CV_Assert(cv::getOptimalDFTSize(hist_len) == (int) hist_len);
  const float df = fps / hist_len;
  for (std::size_t i = 0; i < hist_len/2; i++) {
    freqs.push_back(float(i) * df);
  }

  for (std::size_t i = 0; i < hist_len; i++) {
      hann_window.at<float>(i) = 0.5 * (1.0 - cos( (6.2832f * float(i)) / (float(hist_len) - 1.0)));
      //std::cout << i << " : " << hann_window[i] << std::endl;
  }
  LOG(INFO) << "Hanning window initialized." << hann_window.t();

}

void Periodicity::Update(const cv::Mat &vec, const bool average, const bool verbose) {
  if (average) {
    std::vector<float> temp_power;
    CalcVecDFT(vec, temp_power, hann_window, 0, verbose);
    // You should run addup=false first to fill in fft_power;
    assert(temp_power.size() == fft_power.size() && average_counter >= 1);
    average_counter++;
    for (std::size_t i = 0; i < fft_power.size(); i++)
      // Moving (recursive) average
      fft_power[i] =
          ((float(average_counter - 1) / float(average_counter)) * fft_power[i]) +
          ((1.0/average_counter) * temp_power[i]);
  } else {
    average_counter = 1;
    CalcVecDFT(vec, fft_power, hann_window, 0, verbose);
  }
}


float Periodicity::GetDominantFrequency(const std::size_t start_index) const {
  if (!fft_power.size()) return -1.0;
  CV_Assert(start_index < fft_power.size());

  std::vector<float> fft_power_copy(fft_power);

  // Lazy way to remove low frequency components
  for (std::size_t i = 0; i < start_index; i++) {
    fft_power_copy[i] = 0.0;
  }

  cv::Scalar mean, stddev;
  cv::meanStdDev(fft_power_copy, mean, stddev);
  const double dom_freq_cst = mean[0] + 3.0 * stddev[0];

  float max_power = fft_power_copy.at(0);
  std::size_t max_power_freq_index = 0;

  for (std::size_t i = 1; i < fft_power_copy.size(); i++) {
    if (fft_power_copy.at(i) > max_power) {
      max_power = fft_power_copy.at(i);
      max_power_freq_index = i;
    }
  }

  if (max_power > dom_freq_cst) {
    return float(max_power_freq_index) * fps / hist_len;
  }

  return -1.0;

}
bool CalcVecDFT(const cv::Mat& vec_m,
                std::vector<float>& fft_power,
                const cv::Mat &win_m,
                const unsigned int remove_count,
                const bool verbose)
{
    CV_Assert(vec_m.rows == 1 && win_m.rows == 1 && vec_m.size() == win_m.size());
    std::vector<float> dft_out;
    fft_power.clear();
    if (verbose) {
        LOG(INFO) << "    [FT] Original : " << vec_m;
    }

    cv::Scalar mean = cv::mean(vec_m);
    cv::Mat vec_filtered = (vec_m - mean).mul(win_m);
    cv::blur(vec_filtered, vec_filtered, cv::Size(1, 5));

    if (verbose) {
        LOG(INFO) << "Removed Mean, Windowed & Smoothed: " << vec_filtered;
    }

    cv::dft(vec_filtered, dft_out, cv::DFT_SCALE);

    // TODO: Optimize
    //CCS to Magnitude
    fft_power.resize(vec_filtered.cols / 2);
    fft_power[0] = fabs(dft_out[0]); // sometimes the DC term is negative
    fft_power[vec_filtered.cols/2 - 1] = fabs(dft_out[vec_filtered.cols/2 - 1]);
    unsigned int j = 1;
    for (int i = 1; i < vec_filtered.cols - 2; i+=2) {
        fft_power[j++] = sqrt(dft_out[i] * dft_out[i] + dft_out[i+1] * dft_out[i+1]);
    }

    for (unsigned int i = 0; i < remove_count; i++) {
        fft_power[i] = 0.0;
    }

    if (verbose) {
      LOG(INFO) << "Spectrum: " << cv::Mat(fft_power, false).t();
    }

    return true;
}
