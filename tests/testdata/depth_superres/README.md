Dataset for depth super resolution.
=====

An experimental dataset for qualitative and quantitative evaluation of upsampling algorithms on a synthetic dataset. This work is presented in

    @inproceedings{ferstl2013image,
     title={Image guided depth upsampling using anisotropic total generalized variation},
      author={Ferstl, David and Reinbacher, Christian and Ranftl, Rene and R{\"u}ther, Matthias and Bischof, Horst},
     booktitle={Proceedings of the IEEE International Conference on Computer Vision},
     pages={993--1000},
     year={2013}
    }
    contact: ferstl@icg.tugraz.at

Description: 
==
This dataset is based on Middlebury disparity map. Please consider citing the following:

    @inproceedings{scharstein2003high,
      title={High-accuracy stereo depth maps using structured light},
      author={Scharstein, Daniel and Szeliski, Richard},
     booktitle={2003 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2003. Proceedings.},
      volume={1},
      pages={I--I},
      year={2003},
      organization={IEEE}
    }
    @inproceedings{hirschmuller2007evaluation,
     title={Evaluation of cost functions for stereo matching},
      author={Hirschmuller, Heiko and Scharstein, Daniel},
     booktitle={2007 IEEE Conference on Computer Vision and Pattern Recognition},
     pages={1--8},
     year={2007},
     organization={IEEE}
    }
These images were directly taken from results shared by Jon Barron (Barron JT, Poole B. The fast bilateral solver. In ECCV 2016). For comparison with other methods please see the [dataset](https://drive.google.com/file/d/0B4nuwEMaEsnmaDI3bm5VeDRxams/view).
Jon in turn created this from data obtained from  to J. Park et al. (High Quality Depth Map Upsampling for 3D-TOF Cameras. In ICCV 2011).

Each folder contains the upsampling results from:
- barronpoole	: Barron JT, Poole B. The fast bilateral solver. In ECCV 2016.
- _gth		: hole filled disparity map provided by J. Park et al.
- _input	: raw input images
- bicubic	: used as inputs to barronpoole, dts and hfbs.
- dts		: Bapat and Frahm, The domain transform solver, CVPR 2019.
- hfbs		: A. Mazumdar et al. A hardware-friendly bilateral solver for real-time virtual reality video. In HPG 2017.
- weights	: confidence weights used for the bicubuc target.

In each folder, there are files with following naming rules.
- the number means magnification factor
- 'n' implies input depth is noisy (please see _input folder)
