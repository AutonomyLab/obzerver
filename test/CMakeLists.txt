# Modfied from here: http://stackoverflow.com/a/24228321
include(ExternalProject)
find_package(Threads REQUIRED)

set(GTEST_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/gtest")

ExternalProject_Add(GTestExternal
    URL http://googletest.googlecode.com/files/gtest-1.7.0.zip
    TIMEOUT 10
    PREFIX "${GTEST_PREFIX}"
    INSTALL_COMMAND ""
)

set(LIBPREFIX "${CMAKE_STATIC_LIBRARY_PREFIX}")
set(LIBSUFFIX "${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(GTEST_LOCATION "${GTEST_PREFIX}/src/GTestExternal-build")
set(GTEST_INCLUDE_DIR "${GTEST_PREFIX}/src/GTestExternal/include")
set(GTEST_LIBRARY  "${GTEST_LOCATION}/${LIBPREFIX}gtest${LIBSUFFIX}")

set(GTEST_MAINLIB  "${GTEST_LOCATION}/${LIBPREFIX}gtest_main${LIBSUFFIX}")

add_library(GTest IMPORTED STATIC GLOBAL)
set_target_properties(GTest PROPERTIES
    "IMPORTED_LOCATION" "${GTEST_LIBRARY}"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}")

add_library(GTestMain IMPORTED STATIC GLOBAL)
set_target_properties(GTestMain PROPERTIES
    "IMPORTED_LOCATION" "${GTEST_MAINLIB}"
    "IMPORTED_LINK_INTERFACE_LIBRARIES"
        "${GTEST_LIBRARY};${CMAKE_THREAD_LIBS_INIT}")

add_dependencies(GTest GTestExternal)
include_directories(${GTEST_INCLUDE_DIR})

set(test_SOURCES
  test-cb.cpp
)

add_executable(obz_tests ${test_SOURCES})
target_link_libraries(obz_tests GTestMain)
