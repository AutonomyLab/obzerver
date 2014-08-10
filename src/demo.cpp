#include <iostream>

#include "glog/logging.h"
#include "obzerver/utility.hpp"

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::SetLogDestination(google::GLOG_INFO, "/tmp/demo");
  google::LogToStderr();

  return 0;
}
