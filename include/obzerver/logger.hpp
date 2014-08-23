#ifndef OBZ_LOGGER_H
#define OBZ_LOGGER_H

#include <string>

// Empty log to stderr
// TODO: Configure Level
void obz_log_config(const char* prog_name, const std::string& logfile = std::string(""));

#endif
