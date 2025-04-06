# BundleFit
Bundle Fitting Library for 3D Reconstruction

## Feature
+ Wrapping ceres-solver for bundle adjustment
+ Fast differential calculation by analytical implementation (Partially)
  + available for a standard reprojection operation
+ Pose graph optimization for 6DoF and 7DoF pose representations

## Tutoral slides
[Slide Link](https://speakerdeck.com/opensourceai/tutorial-of-bundle-fitting-library-for-3d-reconstruction-bundlefit)  
[Japanese Version](https://speakerdeck.com/opensourceai/tutorial-of-bundle-fitting-library-for-3d-reconstruction-bundlefit-a2ec77f4-f6c6-46b5-90a9-7c9474d922da)  

## Prerequirements
+ Eigen
+ Ceres Solver
+ OpenCV (if you build demo)
+ nanobind (for Python binding)

## Support
+ Mac OS
+ Ubuntu

Windows only supports builds with Clang.

## Build
Please install dependency.

### Mac OS
Please execute the following command in the terminal.
```bash
brew update
brew install eigen ceres-solver opencv
```

### Ubuntu
Please execute the following command in the terminal.
```bash
sudo apt update
sudo apt install libeigen3-dev libceres-dev libopencv-dev
```

### Build from source
```bash
git clone https://github.com/opensrcai/BundleFit.git
cd BundleFit
mkdir build && cd build
cmake -Dbundlefit_BUILD_DEMO=ON ..
make -j
```

## Run demo
```bash
./demo/demo_corridor_ba
./demo/demo_sim3_point_alignment
```

## Usage
To perform bundle adjustment, you would write the code as follows.

```cpp
template <typename T>
using eigen_std_vector = std::vector<T, Eigen::aligned_allocator<T>>;

  //               .
  //               .
  //               .

  // Shot's pose
  eigen_std_vector<Eigen::Matrix3d> Rs;
  eigen_std_vector<Eigen::Vector3d> ts;

  // 3D points
  eigen_std_vector<Eigen::Vector3d> points;

  // correspondences
  std::vector<std::pair<uint32_t, uint32_t>> correspondences;

  // 2D optimizations
  eigen_std_vector<Eigen::Vector2d> observations;

  //               .
  //               .
  //               .

  // 1. Construct sparse bundle adjuster
  // Please set a second argment to false if you do not use analytic differential. 
  BundleAdjuster bundle_adjuster(1, true);

  // 2. Add perspective camera into bundle_adjuster
  bundle_adjuster.add_perspective_camera(/* camera_id: */ 0, focal, focal, W / 2, H / 2, 0, 
                                         FLAG_FIX_INTRINSIC_PARAMS);

  // 3. Add shot
    // First shot is fixed in true position
  bundle_adjuster.add_SE3_shot(/* id: */ 1, lie::SE3::log(Rs.at(0), ts.at(0)), FLAG_FIX_PARAMS);
    // Second shot is also fixed in true position
  bundle_adjuster.add_SE3_shot(/* id: */ 1, lie::SE3::log(Rs.at(1), ts.at(1)), FLAG_FIX_PARAMS);

    // Add the remaining shots
  for (size_t i = 2; i < num_shots; i++) {
    bundle_adjuster.add_SE3_shot(i, lie::SE3::log(Rs.at(i), ts.at(i)), false);
  }

  // 4. Add point observation
  for (size_t i = 0; i < num_points; i++) {
    const Eigen::Vector3d& point = points.at(i);
    bundle_adjuster.add_landmark(i, point , false);
  }

  for (size_t i = 0; i < num_correspondences; i++) {
    bundle_adjuster.add_reprojection_error(
            0,
            /* shot_id: */ correspondences.at(i).second,
            /* landmark_id: */ correspondences.at(i).first,
            observations.at(i), 1.0F, LossType::TRIVIAL, 1.0
    );
  }
  bundle_adjuster.construct_problem();

  // 5. Execute robust BA
  bundle_adjuster.fit(20);

```

## Python Binding
If you want to use Python bindings, you will need to install the dependencies and set up the Python environment.
Then, please install nanobind by pip.
```bash
pip install nanobind
```
Once nanobind is installed, you can build BundleFit using pip.
```bash
cd /path/to/bundlefit
pip install .
```
Please check if the installation was successful.
```bash
python
>>> import bundlefit
>>> bundlefit.bundlefit_ext.__name__
'bundlefit.bundlefit_ext'
```

Please refer to the Colab example for how to use BundleFit from Python.  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opensrcai/bundlefit/blob/main/example_py/notebooks/bundle_adjustment.ipynb)

## Acknowledgments

This project is made possible by the following open-source projects:

|Project name| License | Functionality |
| :--- | :--- | :--- |
| [Eigen3](https://gitlab.com/libeigen/eigen)| Mozilla Public License 2.0 | Linear Algebra Library |
| [Ceres Solver](https://github.com/ceres-solver/ceres-solver)| The 3-Clause BSD License | Library for Non-linear Least Squares problems |
| [OpenCV](https://github.com/opencv/opencv)| Apache License, Version 2.0 | P3P solver |

## LICENSE

Copyright © 2024 Mikiya Shibuya, Kai Okawa.  
Released under the [Pre-Open Source Verification License](https://opensrcai.org/posvl/posvl-2/).  

Please see the [LICENSE](LICENSE) file for details.
