cmake_minimum_required(VERSION 2.8)
project(obzerver)
find_package(PkgConfig)

set(WITH_TESTS 1 CACHE BOOL "Build Tests (Requires Google Testing Framework)")

set(CMAKE_BUILD_TYPE Debug)

set(obzerver_SOURCES
    src/demo.cpp
    src/utility.cpp
    src/camera_tracker.cpp
)

set(obzerver_HEADERS
    include/obzerver/utility.hpp
    include/obzerver/circular_buffer.hpp
    include/obzerver/camera_tracker.hpp
)

find_package(OpenCV REQUIRED NO_DEFAULT_PATH PATHS /home/autolab/Dev/opencv-2.4.9/build)

include_directories(${obzerver_SOURCE_DIR}/include)

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${OpenCV_LIBS})
    #message(STATUS ${OpenCV_INCLUDE_DIRS})
    #message(STATUS ${LIBS})
endif(OpenCV_FOUND)

# glog debian package is available only on Ubuntu Trusy+
pkg_check_modules(GLOG REQUIRED libglog)
include_directories(${GLOG_INCLUDE_DIRS})
set(LIBS ${LIBS} ${GLOG_LIBRARIES})

message(STATUS "Libs: ${LIBS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ${OpenCV_CXXFLAGS_OTHER} ${GLOG_CXXFLAGS_OTHER}")

add_executable(demo ${obzerver_SOURCES} ${obzerver_HEADERS})

target_link_libraries(demo ${LIBS} )

if(WITH_TESTS)
    add_subdirectory(test)
    enable_testing()
    add_test(NAME Test COMMAND obz_tests)
endif(WITH_TESTS)
