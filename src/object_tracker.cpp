#include "glog/logging.h"

#include "obzerver/object_tracker.hpp"
#include "opencv2/core/core.hpp"

smc_shared_param_t* shared_data;

double ParticleObservationUpdate(long t, const particle_state_t &X)
{
  int NN = 1; // Prevent div/0
  double corr_weight = 0.0;
  for (int i = -9; i < 10; i+=2) {
    for (int j = -9; j < 10; j+=2) {
      int xx = (int) round(X.bb.tl().x) - i;
      int yy = (int) round(X.bb.tl().y) - j;
      if (xx < 30 ||
          yy < 30 ||
          xx > (shared_data->obs_diff.cols - 30) ||
          yy > (shared_data->obs_diff.rows - 30))
      {
        continue;
      }
      NN++;
      corr_weight += shared_data->obs_diff.ptr<uchar>(yy)[xx];
    }
  }
  return fabs(corr_weight) > 1e-12 ? log(corr_weight) : -12.0;
}

smc::particle<particle_state_t> ParticleInitialize(smc::rng *rng)
{
  particle_state_t p;
  p.bb.x = rng->Uniform(shared_data->crop, shared_data->obs_diff.cols - shared_data->crop);
  p.bb.y = rng->Uniform(shared_data->crop, shared_data->obs_diff.rows - shared_data->crop);
  return smc::particle<particle_state_t>(p, -log(shared_data->num_particles)); // log(1/1000)
}

void ParticleMove(long t, smc::particle<particle_state_t> &X, smc::rng *rng)
{
  particle_state_t* cv_to = X.GetValuePointer();
  if (rng->Uniform(0.0, 1.0) > 0.1) {
    cv_to->bb.x += rng->Normal(0, 10);
    cv_to->bb.y += rng->Normal(0, 10);
  } else {
    cv_to->bb.x = rng->Uniform(shared_data->crop, shared_data->obs_diff.cols - shared_data->crop);
    cv_to->bb.y = rng->Uniform(shared_data->crop, shared_data->obs_diff.cols - shared_data->crop);
  }
  X.AddToLogWeight(ParticleObservationUpdate(t, *cv_to));
}

// We are sharing img_diff and img_sof by reference (no copy)
ObjectTracker::ObjectTracker(std::size_t num_particles)
  :num_particles(num_particles),
   crop(30),
   sampler(num_particles, SMC_HISTORY_NONE),
   moveset(ParticleInitialize, ParticleMove),
   ticker(StepBenchmarker::GetInstance())
{
  shared_data = new smc_shared_param_t();
  shared_data->crop = crop;
  shared_data->num_particles = num_particles;
  sampler.SetResampleParams(SMC_RESAMPLE_SYSTEMATIC, 0.5);
  sampler.SetMoveSet(moveset);
  sampler.Initialise();
  LOG(INFO) << "Initialized object tracker with " << num_particles << " particles.";
}

ObjectTracker::~ObjectTracker() {
  delete shared_data;
}

bool ObjectTracker::Update(const cv::Mat &img_diff, const cv::Mat &img_sof)
{
  CV_Assert(img_diff.type() == CV_8UC1);
  //CV_Assert(img_sof.type() == CV_8UC1);

  // This is only shallow copy O(1)
  shared_data->obs_diff = img_diff;
  shared_data->obs_sof = img_sof;

  sampler.Iterate();
  ticker.tick("  [OT] Particle Filter");
  return true;
}


/* Debug */

void ObjectTracker::DrawParticles(cv::Mat &img)
{
  for (long i = 0; i < sampler.GetNumber(); i++) {
    cv::circle(img, sampler.GetParticleValue(i).bb.tl(), std::min(0.0, sampler.GetParticleWeight(i)), cv::Scalar(0, 0, 0));
  }
}
