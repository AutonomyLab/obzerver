cmake_minimum_required(VERSION 2.8)
project(obzerver)
find_package(PkgConfig)

set(PROJECT_VERSION_MAJOR "0")
set(PROJECT_VERSION_MINOR "1")
set(PROJECT_VERSION_PATCH "0")
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

set(WITH_TESTS 0 CACHE BOOL "Build Tests (Requires Google Testing Framework)")
#set(USE_CUSTOM_OPENCV 0 CACHE BOOL "Use Custom Path for OpenCV (Provided by OpenCV_DIR")

# This requires downloading and compiling two external deps
set(WITH_PF_CCV_TRACKER 0 CACHE BOOL "Build Particle Filter and Object Detection Based Single Object Tracker")

set(obzerver_SOURCES
    src/utility.cpp
    src/benchmarker.cpp
    src/camera_tracker.cpp
    src/self_similarity.cpp
    src/fft.cpp
    src/tobject.cpp
    src/logger.cpp    
    src/roi_extraction.cpp
    src/hungarian.cpp
    src/kalman.cpp
    src/multi_object_tracker.cpp
    src/periodicity_app.cpp
)

set(obzerver_HEADERS
    include/obzerver/common_types.hpp
    include/obzerver/utility.hpp
    include/obzerver/benchmarker.hpp
    include/obzerver/circular_buffer.hpp
    include/obzerver/camera_tracker.hpp
    include/obzerver/self_similarity.hpp
    include/obzerver/fft.hpp
    include/obzerver/tobject.hpp
    include/obzerver/logger.hpp
    include/obzerver/opencv3-backport/shift.hpp
    include/obzerver/roi_extraction.hpp
    include/obzerver/hungarian.hpp
    include/obzerver/kalman.hpp
    include/obzerver/multi_object_tracker.hpp
    include/obzerver/periodicity_app.hpp
)

#if(USE_CUSTOM_OPENCV)
#  message(STATUS "Using custom OpenCV path: ${OpenCV_DIR}")
#  find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
#else(USE_CUSTOM_OPENCV)
#  message(STATUS "Using default OpenCV path.")
#  find_package(OpenCV REQUIRED)
#endif(USE_CUSTOM_OPENCV)

include_directories(${obzerver_SOURCE_DIR}/include)

find_package(OpenCV 2.4 REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${OpenCV_LIBS})
    message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV CXXFLAGS: ${OpenCV_CXXFLAGS_OTHER}")
    message(STATUS  ${OpenCV_LIBS})
endif(OpenCV_FOUND)

## glog debian package is available only on Ubuntu Trusy+
pkg_check_modules(GLOG REQUIRED libglog)
include_directories(${GLOG_INCLUDE_DIRS})
set(LIBS ${LIBS} ${GLOG_LIBRARIES})

## Boost (for program options)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

# -march=corei7-avx -mtune=corei7-avx -mno-avx -mno-aes
# -fopenmp is because of upstram dbscan, fix it
set(CMAKE_OPT_FLAGS "-march=native -ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_OPT_FLAGS} ${CMAKE_CXX_FLAGS} ${OpenCV_CXXFLAGS_OTHER} ${GLOG_CXXFLAGS_OTHER} -std=c++11 -Wall -Wextra -fopenmp -march=native")

## Build and Configure External Dependencies (Modified)

include(ExternalProject)
set(DEVEL_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/install")
include_directories(${DEVEL_INSTALL_DIR}/include)
message(STATUS "Devel install directory for external deps: ${DEVEL_INSTALL_DIR}")

if(WITH_PF_CCV_TRACKER)
  message(STATUS "Building Particle Filter based Single Object Tracker (w/ ccv and smctc)")
  set(obzerver_SOURCES ${obzerver_SOURCES}
                       src/object_tracker.cpp
                       src/ccv_wrapper.cpp
  )

  set(obzerver_HEADERS ${obzerver_HEADERS}
                       include/obzerver/object_tracker.hpp
                       include/obzerver/ccv_wrapper.hpp
  )

  add_executable(ccv_demo src/ccv_demo.cpp)
  target_link_libraries(ccv_demo obzerver ${OpenCV_LIBS} ${GLOG_LIBRARIES})

  # libsmctc
  ExternalProject_Add(libsmctc
    GIT_REPOSITORY https://github.com/mani-monaj/smctc.git
    GIT_TAG master
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}
    CMAKE_ARGS
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX=${DEVEL_INSTALL_DIR}
    INSTALL_COMMAND make install
    BUILD_IN_SOURCE 0
  )
  set(LIBS ${LIBS} "${DEVEL_INSTALL_DIR}/lib/libsmctc.so")

  # libccv's build system is convoluted
  ExternalProject_Add(libccv_project
    URL https://github.com/mani-monaj/ccv-lean/archive/stable.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}
    CONFIGURE_COMMAND cd lib && ./configure --with-cuda=no --prefix=${DEVEL_INSTALL_DIR}
    BUILD_COMMAND cd lib && make
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
  )

  include_directories(${CMAKE_CURRENT_BINARY_DIR}/src/libccv_project/lib)

  # We also need to take care of libccv's transitive dependencies (png, jpeg and dispatch, more might be needed)
  set(LIBS ${LIBS} "${CMAKE_CURRENT_BINARY_DIR}/src/libccv_project/lib/libccv.a;dispatch;jpeg;png;BlocksRuntime;gsl;gslcblas")
endif(WITH_PF_CCV_TRACKER)


# dbscan
ExternalProject_Add(libdbscan_project
  GIT_REPOSITORY https://github.com/mani-monaj/DBSCAN.git
  GIT_TAG master
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  CMAKE_ARGS
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${DEVEL_INSTALL_DIR}
    -DBUILD_PYTHONLIB=OFF
    -DWITH_OPENMP=ON
  INSTALL_COMMAND make install
  BUILD_IN_SOURCE 0
)

set(LIBS ${LIBS} "${DEVEL_INSTALL_DIR}/lib/libdbscan.so")

## Debug

message(STATUS "Libs: ${LIBS}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# Since libccv is a static library, wee need to compile obzerver as static as well
add_library(obzerver STATIC ${obzerver_SOURCES} ${obzerver_HEADERS})
target_link_libraries(obzerver ${LIBS})
add_dependencies(obzerver libsmctc libccv_project libdbscan_project)

add_executable(demo src/demo.cpp)
target_link_libraries(demo obzerver ${OpenCV_LIBS} ${GLOG_LIBRARIES})

if(WITH_TESTS)
    add_subdirectory(test)
    enable_testing()
    add_test(NAME Test COMMAND obz_tests)
endif(WITH_TESTS)

# Without proper find-obzerver.cmake, the library wont work with isolated
# catkin workspaces.
# The cmake.in file has been borrowed from mavlink project

set(PKG_NAME ${PROJECT_NAME})
set(PKG_VERSION ${PROJECT_VERSION})
set(PKG_DESC "obzerver")
set(PKG_LIBRARIES )
set(PKG_DEPENDS )

configure_file(config.cmake.in ${PROJECT_NAME}-config.cmake @ONLY)
install(
  FILES ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
  DESTINATION share/${PROJECT_NAME}/cmake/ COMPONENT cmake
)

install(TARGETS obzerver
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(FILES ${obzerver_HEADERS}
  DESTINATION include/obzerver
)

install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)
