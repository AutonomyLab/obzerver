cmake_minimum_required(VERSION 2.8)
project(obzerver)
find_package(PkgConfig)

set(WITH_TESTS 1 CACHE BOOL "Build Tests (Requires Google Testing Framework)")

set(CMAKE_BUILD_TYPE Release)

set(obzerver_SOURCES
    src/demo.cpp
    src/utility.cpp
    src/benchmarker.cpp
    src/camera_tracker.cpp
    src/object_tracker.cpp
    src/self_similarity.cpp
)

set(obzerver_HEADERS
    include/obzerver/utility.hpp
    include/obzerver/benchmarker.hpp
    include/obzerver/circular_buffer.hpp
    include/obzerver/camera_tracker.hpp
    include/obzerver/object_tracker.hpp
    include/obzerver/self_similarity.hpp
)

find_package(OpenCV REQUIRED NO_DEFAULT_PATH PATHS /home/autolab/Dev/opencv-2.4.9/build)

include_directories(${obzerver_SOURCE_DIR}/include)

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${OpenCV_LIBS})
    #message(STATUS ${OpenCV_INCLUDE_DIRS})
    #message(STATUS ${LIBS})
endif(OpenCV_FOUND)

#find_package(MRPT REQUIRED base bayes)
#include_directories(${MRPT_INCLUDE_DIRS})
#set(LIBS ${LIBS} ${MRPT_LIBS})

# glog debian package is available only on Ubuntu Trusy+
pkg_check_modules(GLOG REQUIRED libglog)
include_directories(${GLOG_INCLUDE_DIRS})
set(LIBS ${LIBS} ${GLOG_LIBRARIES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ${OpenCV_CXXFLAGS_OTHER} ${GLOG_CXXFLAGS_OTHER} -std=c++0x")
add_subdirectory("3rdparty/smctc")
include_directories(${smctc_SOURCE_DIR}/include)
set(LIBS ${LIBS} "smctc")

message(STATUS "Libs: ${LIBS}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")

add_executable(demo ${obzerver_SOURCES} ${obzerver_HEADERS})

target_link_libraries(demo ${LIBS} )

if(WITH_TESTS)
    add_subdirectory(test)
    enable_testing()
    add_test(NAME Test COMMAND obz_tests)
endif(WITH_TESTS)
