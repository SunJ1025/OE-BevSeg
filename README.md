# Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?

This is the official code release for our paper 
**OE-BevSeg: An Object Informed and Environment Aware Multimodal Framework for Bird’s-eye-view Vehicle Semantic Segmentation. **

[[Paper](https://arxiv.org/abs/2407.13137)] [[Project Page](https://github.com/SunJ1025/OE-BevSeg/)]

<img src='https://github.com/SunJ1025/OE-BevSeg/blob/main/videos/output_07.gif'>

<img src='https://github.com/SunJ1025/OE-BevSeg/blob/main/videos/output_39.gif'>

## Requirements

The lines below should set up a fresh environment with everything you need: 
```
conda create --name bev
source activate bev 
conda install pytorch=1.12.0 torchvision=0.13.0 cudatoolkit=11.3 -c pytorch
conda install pip
pip install -r requirements.txt
```

## Training

A sample training command is included in `train.sh`.


## Evaluation

A sample evaluation command is included in `eval.sh`.


## Acknowledgements

We would like to express our gratitude to the authors of the following codebase.

## [[simplebev](https://github.com/SunJ1025/OE-BevSeg/)]


