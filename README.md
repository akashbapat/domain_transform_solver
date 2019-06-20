## Domain Transform Solver
This code is not the exact version used in the paper, but has undergone many changes.
### About
This repository "Domain Transform Solver" is the CUDA implementation for paper "The Domain Transform Solver" published in CVPR 2019. This software is licensed under the BSD 3-Clause license.
If you use this project for your research, please cite:

    @inproceedings{bapat2019dts,
      title={The Domain Transform Solver},
      author={Bapat, Akash and Frahm, Jan-Michael},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      year={2019},
    }

### Compilation
You need the following dependencies to run the code.

1. Cuda 8 or 9. Does not work with Cuda 10. 
2. OpenCV
    - Mainly used for image IO or in the *cli.cpp, not used in the main optimization code.
3. GFlags
    - Only needed to process the commandline flags. The code is organized in such a way to simply remove the flags at the top of each *_cli.cpp with your favorite cmd flag processor. One of the favorites is Boost.
4. GTest (optional)
    - Turn off tests if GTest is not available. Not so important for using DTS.
5. LibCua (https://github.com/trueprice/libcua)
    - Automatically downloaded.

#### Installing dependencies 
I assume a familiarity with cmake and CMakeLists. The code was tested with CMake 3.5.1
1. Installing OpenCV
   ```
   sudo apt-get install libopencv-dev
   ```
   
2. Installing Glags:
   ```
   git clone https://github.com/gflags/gflags.git
   cd gflags
   mkdir build 
   cd build
   cmake ..
   make -j4
   sudo make install
   ```
   
3. Installing Gtest:
   ```
   git clone https://github.com/google/googletest.git
   cd googletest
   mkdir build 
   cd build
   cmake ..
   make -j4
   sudo make install
   ```
#### Installing DTS
This repository is setup with submodules which need to be also clones using the recursive flag as follows:
   ```
   git clone --recursive https://github.com/akashbapat/domain_transform_solver.git
   cd domain_transform_solver
   mkdir build
   cd build
   cmake ..
   make
   ```
   
### Running the solver via the commandline interfaces (CLIs)
Once you have built the code you can run the solver examples as follows. When inside the folder domain_transform_solver/build
1. Stereo solver
   ```
   ./src/benchmark_stereo_cli --path_to_data_top_dir ../tests/testdata/stereo/
   ```
2. Colorization solver
 Change the algorithm_name to "dts" to use  domain transform solver.
   ```
   ./src/benchmark_colorization_cli --path_to_data_top_dir ../tests/testdata/colorization/ --algorithm_name hfbs
   ```
   
3. Depth super resolution solver
Change the algorithm_name to "hfbs" to use Mazumdar et al, Hardware friendly bilateral solver.
   ```
   ./src/benchmark_depth_super_res_cli --path_to_data_top_dir ../tests/testdata/depth_superres/ --algorithm_name dts
   ```
### Contribution
Feel free to create pull requests if you have a contribution or bugfixes.

### Contact
akash@cs.unc.edu
