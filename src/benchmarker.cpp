#include <sstream>
#include <iostream>
#include <iomanip>

#include "obzerver/benchmarker.hpp"
#include "opencv2/core.hpp"

StepBenchmarker::StepBenchmarker(): last_tick(cv::getTickCount()) {
    reset();
}

StepBenchmarker& StepBenchmarker::GetInstance() {
  static StepBenchmarker instance;
  return instance;
}

void StepBenchmarker::reset() {
    items.clear();
    items.push_back(std::make_pair("Start", 0.0));
}

void StepBenchmarker::tick() {
    std::stringstream stream;
    stream << "Unknown " << items.size();
    items.push_back(std::make_pair(stream.str(), update()));
}

void StepBenchmarker::tick(const std::string& text) {
    items.push_back(std::make_pair(text, update()));
}

std::string StepBenchmarker::getstr(const bool clear_screen) const {
  std::stringstream stream;
  double total = 0;
  for (unsigned int i = 0; i < items.size(); i++) {
    stream << std::setw(30) << std::left
           << items[i].first
           << std::setw(10)
           << items[i].second << " (ms)"
           << std::endl;
    total += items[i].second;
  }
  stream << std::setw(30) << std::left
         << "[Total]" << std::setw(10) << total << " (ms)" << std::endl;
  if (clear_screen) {
    // From: http://stackoverflow.com/q/4062045/1215297
    stream << "\033[2J\033[1;1H";
  }
  return stream.str();
}

void StepBenchmarker::dump(const bool clear_screen) const {
  std::cerr << getstr(clear_screen);
}

double StepBenchmarker::update() {
    long int now = cv::getTickCount();
    const double diff = 1000.0 * ((double) (now - last_tick) / (double) cv::getTickFrequency());
    last_tick = now;
    return diff;
}
