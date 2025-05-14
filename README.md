# WRF-GS+
Welcome to the WRF-GS+ repository, an enhanced version of the original WRF-GS. You can access the accompanying paper, ["Neural Representation for Wireless Radiation Field Reconstruction: A 3D Gaussian Splatting Approach"](https://arxiv.org/abs/2412.04832v4), for more details.

Abstract: Wireless channel modeling plays a pivotal role in designing, analyzing, and optimizing wireless communication systems. Nevertheless, developing an effective channel modeling approach has been a long-standing challenge. This issue has been escalated due to denser network deployment, larger antenna arrays, and broader bandwidth in next-generation networks. To address this challenge, we put forth WRF-GS, a novel framework for channel modeling based on wireless radiation field (WRF) reconstruction using 3D Gaussian splatting (3D-GS). WRF-GS employs 3D Gaussian primitives and neural networks to capture the interactions between the environment and radio signals, enabling efficient WRF reconstruction and visualization of the propagation characteristics. The reconstructed WRF can then be used to synthesize the spatial spectrum for comprehensive wireless channel characterization. While WRF-GS demonstrates remarkable effectiveness, it faces limitations in capturing high-frequency signal variations caused by complex multipath effects. To overcome these limitations, we propose WRF-GS+, an enhanced framework that integrates electromagnetic wave physics into the neural network design. WRF-GS+ leverages deformable 3D Gaussians to model both static and dynamic components of the WRF, significantly improving its ability to characterize signal variations. In addition, WRF-GS+ enhances the splatting process by simplifying the 3D-GS modeling process and improving computational efficiency. Experimental results demonstrate that both WRF-GS and WRF-GS+ outperform baselines for spatial spectrum synthesis, including ray tracing and other deep-learning approaches.

## Installation
Create the basic environment
```python
conda env create --file environment.yml
conda activate wrfgsplus
```
Install some extensions
```python
cd submodules
pip install ./simple-knn
pip install ./diff-gaussian-rasterization # or cd ./diff-gaussian-rasterization && python setup.py develop
pip install ./fused-ssim
```

## Training & Evaluation
Due to file size limitations, a small dataset is included to help quickly verify the code, which can be executed using the following command:
```python
python train.py
```
More datasets can be found [here](https://github.com/XPengZhao/NeRF2?tab=readme-ov-file).<be>

## To-Do List
- [ ] Release more case study code.
- [ ] Optimize related code structure.

## BibTex
If you find this work useful in your research, please cite:
```bibtex
@article{wen2025wrfgsplus,
  title={Neural Representation for Wireless Radiation Field Reconstruction: A 3D Gaussian Splatting Approach},
  author={Wen, Chaozheng and Tong, Jingwen and Hu, Yingdong and Lin, Zehong and Zhang, Jun},
  journal={arXiv preprint arXiv:2412.04832v3},
  year={2025}
}
```
## Acknowledgment
Some code snippets are borrowed from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting/tree/main).
