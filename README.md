## Pre-requirements

- Optimized OpenCV
- Google glog: libgoogle-glog-dev
- GNU Scientific Library: libgsl0-dev
- git (for cmake external project)

## Compile Instructions

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/PATH/TO/OPENCV/INSTALL -DCMAKE_BUILD_TYPE=RelWithDebInfo
```
