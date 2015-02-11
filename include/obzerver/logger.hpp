#ifndef OBZ_LOGGER_H
#define OBZ_LOGGER_H

#include <string>

namespace obz
{

// Empty log to stderr
// TODO: Configure Level
void log_config(const char* prog_name, const std::string& logfile = std::string(""));

}  // namespace obz
#endif
