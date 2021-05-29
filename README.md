
# Update to Python 3

As the original was intended for Python 2 and a lot of libraries have become outdated in the meantime, this fork intends to use the code for Python 3 with newer versions of PyTorch such that e.g. CUDA 11 is supported as well. 

We have provided an Anaconda environment, but feel free to install the dependencies as you wish. Especially Pytorch, make sure you have the cuda version that fits your GPU. By default we have `pytorch=1.8.1=py3.8_cuda11.1_cudnn8.0.5_0`.

This should run everything without Docker, relative paths are used so run `main.py` and `live_visualization_custom.py` in the repository of `/dense_correspondence`. Similarly, when downloading the data, by default the paths are configured to point to `/Data` in the main repository (`/` is this repository directory).

The use of YAML files is quite ubiquitous in the original repository, while this fork intends to keep editing these YAML files to a minimum, we need to adjust one of them while evaluating our trained network: `config/dense_correspondence/evaluation/evaluation.yaml`. This one has "path_to_network_params : trained_models/tutorials/caterpillar_3/003501.pth" as a key which is hard coded. The "003501" stands for the number of iterations your network was trained for. By default this should indeed be 3500 (in "main.py"), but this should be adjusted in both if you wish to train for more or less iterations.

After training, testing on some custom images can be done with "live_visualization_custom.py", where in line 104 - 105 you can pick a pair of images that we will visualize and can interactively see point correspondence with, using the mouse in the first image (OpenCV windows will pop up when running this file). This is by no means a clean way of doing it, but this is how you can test what you have.


### NOTE: not all files have been checked for Python 3 compatibility yet, but this should mostly be print statements that need to be adjusted.


## Getting it working

More or less the same as the original, just without Docker:
### Step 1
```
git clone https://github.com/mmichelis/dense-object-nets-python3.git
```

### Step 2
This only gets a bit of the data, things will be different if you want all data.
```
# navigate to the root of the project so paths can be inferred
cd pytorch-dense-correspondence
python config/download_pdc_data.py config/dense_correspondence/dataset/composite/caterpillar_upright.yaml ./Data
```

### Step 3
```
git submodule update --init --recursive
```

### Step 4
Lastly we get the required environment packages. Anaconda is the most convenient, but you can do it without as well. The most important packages are just numpy, scipy and pytorch. There might be a few more, but you'll have to install them as the errors appear while running the code.
```
conda create -f environment.yml
```






# Original README starts here
### Updates 

- September 4, 2018: Tutorial and data now available!  [We have a tutorial now available here](./doc/tutorial_getting_started.md), which walks through step-by-step of getting this repo running.
- June 26, 2019: We have updated the repo to pytorch 1.1 and CUDA 10. For code used for the experiments in the paper see [here](https://github.com/RobotLocomotion/pytorch-dense-correspondence/releases/tag/pytorch-0.3).


## Dense Correspondence Learning in PyTorch

In this project we learn Dense Object Nets, i.e. dense descriptor networks for previously unseen, potentially deformable objects, and potentially classes of objects:

![](./doc/caterpillar_trim.gif)  |  ![](./doc/shoes_trim.gif) | ![](./doc/hats_trim.gif)
:-------------------------:|:-------------------------:|:-------------------------:

We also demonstrate using Dense Object Nets for robotic manipulation tasks:

![](./doc/caterpillar_grasps.gif)  |  ![](./doc/shoe_tongue_grasps.gif)
:-------------------------:|:-------------------------:

### Dense Object Nets: Learning Dense Visual Descriptors by and for Robotic Manipulation

This is the reference implementation for our paper:

[PDF](https://arxiv.org/pdf/1806.08756.pdf) | [Video](https://www.youtube.com/watch?v=L5UW1VapKNE)

[Pete Florence*](http://www.peteflorence.com/), [Lucas Manuelli*](http://lucasmanuelli.com/), [Russ Tedrake](https://groups.csail.mit.edu/locomotion/russt.html)

<em><b>Abstract:</b></em> What is the right object representation for manipulation? We would like robots to visually perceive scenes and learn an understanding of the objects in them that (i) is task-agnostic and can be used as a building block for a variety of manipulation tasks, (ii) is generally applicable to both rigid and non-rigid objects, (iii) takes advantage of the strong priors provided by 3D vision, and (iv) is entirely learned from self-supervision.  This is hard to achieve with previous methods: much recent work in grasping does not extend to grasping specific objects or other tasks, whereas task-specific learning may require many trials to generalize well across object configurations or other tasks.  In this paper we present Dense Object Nets, which build on recent developments in self-supervised dense descriptor learning, as a consistent object representation for visual understanding and manipulation. We demonstrate they can be trained quickly (approximately 20 minutes) for a wide variety of previously unseen and potentially non-rigid objects.  We additionally present novel contributions to enable multi-object descriptor learning, and show that by modifying our training procedure, we can either acquire descriptors which generalize across classes of objects, or descriptors that are distinct for each object instance. Finally, we demonstrate the novel application of learned dense descriptors to robotic manipulation. We demonstrate grasping of specific points on an object across potentially deformed object configurations, and demonstrate using class general descriptors to transfer specific grasps across objects in a class. 

#### Citing

If you find this code useful in your work, please consider citing:

```
@article{florencemanuelli2018dense,
  title={Dense Object Nets: Learning Dense Visual Object Descriptors By and For Robotic Manipulation},
  author={Florence, Peter and Manuelli, Lucas and Tedrake, Russ},
  journal={Conference on Robot Learning},
  year={2018}
}
```

### Tutorial

- [getting started with pytorch-dense-correspondence](./doc/tutorial_getting_started.md)

### Code Setup

- [setting up docker image](doc/docker_build_instructions.md)
- [recommended docker workflow ](doc/recommended_workflow.md)

### Dataset

- [data organization](doc/data_organization.md)
- [data pre-processing for a single scene](doc/data_processing_single_scene.md)

### Training and Evaluation
- [training a network](doc/training.md)
- [evaluating a trained network](doc/dcn_evaluation.md)
- [pre-trained models](doc/model_zoo.md)

### Miscellaneous
- [coordinate conventions](doc/coordinate_conventions.md)
- [testing](doc/testing.md)

### Git management

To prevent the repo from growing in size, recommend always "restart and clear outputs" before committing any Jupyter notebooks.  If you'd like to save what your notebook looks like, you can always "download as .html", which is a great way to snapshot the state of that notebook and share.
