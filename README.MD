# DOPE: Distillation Of Part Experts for whole-body 3D pose estimation in the wild

This repository contains the code for running our DOPE model.
We only provide code for testing, not for training.
If you use our code, please cite our [ECCV'20 paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710375.pdf):

```bibtex
@inproceedings{dope,
  title={{DOPE: Distillation Of Part Experts for whole-body 3D pose estimation in the wild}},
  author={{Weinzaepfel, Philippe and Br\'egier, Romain and Combaluzier, Hadrien and Leroy, Vincent and Rogez, Gr\'egory},
  booktitle={{ECCV}},
  year={2020}
}
```

## License

DOPE is distributed under the CC BY-NC-SA 4.0 License. See [LICENSE](LICENSE) for more information.

### Getting started

Our python3 code requires the following packages:
* pytorch
* torchvision
* opencv (for drawing the results)
* numpy/scipy

Our code has been tested on Linux, with pytorch 1.5 and torchvision 0.6.
We do not provide support for installation.

#### Download the models

First create a folder `models/` in which you should place the downloaded pretrained models.
The list of models include:
* [DOPE_v1_0_0](http://download.europe.naverlabs.com/ComputerVision/DOPE_models/DOPE_v1_0_0.pth.tgz) as used in our ECCV'20 paper
* [DOPErealtime_v1_0_0](http://download.europe.naverlabs.com/ComputerVision/DOPE_models/DOPErealtime_v1_0_0.pth.tgz) which is its real-time version

#### post-processing with a modified version of LCR-Net++

Our post-processing relies on a modified version of the pose proposals integration proposed in the [LCR-Net++ code](https://thoth.inrialpes.fr/src/LCR-Net/).
To get this code, once in the DOPE folder, please clone our modified LCR-Net++ repository:
```
git clone https://github.com/naver/lcrnet-v2-improved-ppi.git
```

Alternatively, you can use a more naive post-processing based on non-maximum suppression by add `--postprocess nms` to the commandlines below, which will result in a slight decrease of performance.


## Using the code

To use our code on an image, use the following command:

```
python dope.py --model <modelname> --image <imagename>
```

with
* `<modelname>`: name of model to use (eg DOPE_v1_0_0)
* `<imagename>`: name of the image to test

For instance, you can run
```
python dope.py  --model DOPErealtime_v1_0_0 --image 015994080.jpg
```

The command will create an image `<imagename>_<modelname>.jpg` that shows the 2D poses output by our DOPE model.

We also provide code to visualize the results but in 2D and in 3D, just add the argument `--visu3d` to the previous commandline.
The 3D visualization uses OpenGL and the visvis python package, we do not provide support for installation and OpenGL issues.
Running the command with the `--visu3d` option should create another file with the 3D visualization named `<imagename>_<modelname>_visu3d.jpg`.
Note that DOPE predicts 3D poses for each body part in their relative coordinate systems, eg, centered on body center for bodies.
For better visualization, we approximate 3D scene coordinates by finding the offsets that minimize the reprojection error based on a scaled orthographic projection model.

Here is one example resulting image:
![example result](example_result.jpg)

Our real-time models use half computation. In case your device cannot handle it, please uncomment the line `  #ckpt['half'] = False` in `dope.py`.


