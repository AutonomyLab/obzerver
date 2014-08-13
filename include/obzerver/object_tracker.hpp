#ifndef OBJECT_TRACKER_HPP
#define OBJECT_TRACKER_HPP

#include "smctc/smctc.hh"
#include "opencv2/core/core.hpp"
#include "obzerver/benchmarker.hpp"

struct smc_shared_param_t {
  cv::Mat obs_diff;
  cv::Mat obs_sof;
  unsigned short int crop;
  std::size_t num_particles;
};

extern smc_shared_param_t* shared_data;

struct particle_state_t {
  cv::Rect_<double> bb;
};

double ParticleObservationUpdate(long t, const particle_state_t& X);
smc::particle<particle_state_t> ParticleInitialize(smc::rng* rng);
void ParticleMove(long t, smc::particle<particle_state_t> &X, smc::rng* rng);

class ObjectTracker {
protected:
  std::size_t num_particles;
  unsigned short int crop;

  smc::sampler<particle_state_t> sampler;
  smc::moveset<particle_state_t> moveset;

  StepBenchmarker& ticker;
public:
  ObjectTracker(const std::size_t num_particles);
  ~ObjectTracker();

  bool Update(const cv::Mat& img_diff, const cv::Mat& img_sof);
  void DrawParticles(cv::Mat& img);
};

#endif
