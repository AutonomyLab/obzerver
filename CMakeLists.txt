cmake_minimum_required(VERSION 2.8)
project(obzerver)
find_package(PkgConfig)

set(PROJECT_VERSION_MAJOR "0")
set(PROJECT_VERSION_MINOR "1")
set(PROJECT_VERSION_PATCH "0")
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

set(WITH_TESTS 0 CACHE BOOL "Build Tests (Requires Google Testing Framework)")
set(USE_CUSTOM_OPENCV 0 CACHE BOOL "Use Custom Path for OpenCV (Provided by OpenCV_DIR")

set(obzerver_SOURCES
    src/utility.cpp
    src/benchmarker.cpp
    src/camera_tracker.cpp
    src/object_tracker.cpp
    src/self_similarity.cpp
    src/fft.cpp
    src/tobject.cpp
)

set(obzerver_HEADERS
    include/obzerver/utility.hpp
    include/obzerver/benchmarker.hpp
    include/obzerver/circular_buffer.hpp
    include/obzerver/camera_tracker.hpp
    include/obzerver/object_tracker.hpp
    include/obzerver/self_similarity.hpp
    include/obzerver/fft.hpp
    include/obzerver/tobject.hpp
    include/obzerver/opencv3-backport/shift.hpp
)

if(USE_CUSTOM_OPENCV)
  message(STATUS "Using custom OpenCV path: ${OpenCV_DIR}")
  find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
else(USE_CUSTOM_OPENCV)
  message(STATUS "Using default OpenCV path.")
  find_package(OpenCV REQUIRED)
endif(USE_CUSTOM_OPENCV)

include_directories(${obzerver_SOURCE_DIR}/include)

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${OpenCV_LIBS})
    message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")
    #message(STATUS ${LIBS})
endif(OpenCV_FOUND)

# glog debian package is available only on Ubuntu Trusy+
pkg_check_modules(GLOG REQUIRED libglog)
include_directories(${GLOG_INCLUDE_DIRS})
set(LIBS ${LIBS} ${GLOG_LIBRARIES})

# -march=corei7-avx -mtune=corei7-avx -mno-avx -mno-aes
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenCV_CXXFLAGS_OTHER} ${GLOG_CXXFLAGS_OTHER} -std=c++0x -Wall -Wextra")
add_subdirectory("3rdparty/smctc")
include_directories(${smctc_SOURCE_DIR}/include)
set(LIBS ${LIBS} "smctc")

message(STATUS "Libs: ${LIBS}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

add_library(obzerver SHARED ${obzerver_SOURCES} ${obzerver_HEADERS})
target_link_libraries(obzerver ${LIBS})

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
