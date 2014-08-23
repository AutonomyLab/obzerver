#include <iostream>

#include "obzerver/logger.hpp"
#include "glog/logging.h"


void obz_log_config(const char *prog_name, const std::string& logfile) {
  google::InitGoogleLogging(prog_name);
  if (logfile.empty()) {
    google::LogToStderr();
  } else {
    std::cout << "[OBZERVER] Logging to: " <<  logfile << std::endl;
    google::SetLogDestination(google::GLOG_INFO, logfile.c_str());
  }
}
