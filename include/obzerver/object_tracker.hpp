#ifndef OBJECT_TRACKER_HPP
#define OBJECT_TRACKER_HPP

#include "smctc/smctc.hh"
#include "opencv2/core/core.hpp"
#include "obzerver/benchmarker.hpp"
#include "obzerver/circular_buffer.hpp"

/* SMC Stuff */

struct smc_shared_param_t {
  cv::Mat obs_diff;
  cv::Mat obs_sof;
  unsigned short int crop;
  std::size_t num_particles;
  double prob_random_move;
  double mm_displacement_stddev;
};

extern cv::Ptr<smc_shared_param_t> shared_data;

// It's very important that every element of this
// structure be of type double.
// because the total number of elements of
// smc::particle<particle_state_t> is being calculated
// assuming that all elements are of size double.
// creating a cv::Mat wrapper around particles needs
// homogenous elements in the mentioned data structure
struct particle_state_t {
  cv::Rect_<double> bb;
  bool recent_random_move;
};

double ParticleObservationUpdate(long t, const particle_state_t& X);
smc::particle<particle_state_t> ParticleInitialize(smc::rng* rng);
void ParticleMove(long t, smc::particle<particle_state_t> &X, smc::rng* rng);

/* The Tracker */

class TObject {
public:
  cv::Rect bb;
  std::size_t age;

  TObject(): age(0) {;}
  TObject(const cv::Rect& r): bb(r), age(0) {;}
  TObject(const cv::Rect &r, const std::size_t age): bb(r), age(age) {;}
};

enum tracking_status_t {
  TRACKING_STATUS_LOST = 0,
  TRACKING_STATUS_NEW,
  TRACKING_STATUS_TRACKING,
  TRACKING_STATUS_DISAPPEAR,
  TRACKING_STATUS_NUM
};

class ObjectTracker {
protected:
  std::size_t num_particles;
  std::size_t hist_len;
  unsigned short int crop; //px
  double prob_random_move; //[0,1]
  double mm_displacement_noise_stddev; //px

  smc::sampler<particle_state_t> sampler;
  smc::moveset<particle_state_t> moveset;

  // Clustering and Object Tracking
  tracking_status_t status;
  std::size_t tracking_counter;
  unsigned short int num_clusters;
  cv::Mat labels;
  cv::Mat centers;
  double clustering_err_threshold;
  CircularBuffer<TObject> object_hist;

  StepBenchmarker& ticker;

  cv::Rect GenerateBoundingBox(const std::vector<cv::Point2f> &pts, const float alpha, const int boundary_width, const int boundary_height);
public:
  ObjectTracker(const std::size_t num_particles,
                const std::size_t hist_len,
                const unsigned short int crop = 30,
                const double prob_random_move = 0.2,
                const double mm_displacement_noise_stddev = 5);
  ~ObjectTracker();

  bool Update(const cv::Mat& img_diff, const cv::Mat& img_sof);

  cv::Rect GetBoundingBox(unsigned int t = 0) const;
  void DrawParticles(cv::Mat& img);

};

#endif
