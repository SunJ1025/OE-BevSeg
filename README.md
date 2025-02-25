# Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?

This is the official code release for our paper 
**OE-BevSeg: An Object Informed and Environment Aware Multimodal Framework for Bird’s-eye-view Vehicle Semantic Segmentation. **

[[Paper](https://arxiv.org/abs/2407.13137)] [[Project Page](https://github.com/SunJ1025/OE-BevSeg/)]

<img src='https://simple-bev.github.io/videos/output_compressed.gif'>



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



## Citation

If you use this code for your research, please cite:


Bibtex:
```
@inproceedings{harley2022simple,
  title={Simple-{BEV}: What Really Matters for Multi-Sensor BEV Perception?},
  author={Adam W. Harley and Zhaoyuan Fang and Jie Li and Rares Ambrus and Katerina Fragkiadaki},
  booktitle={arXiv:2206.07959},
  year={2022}
}
```
