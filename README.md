# AdmReg

Pytorch code for the paper:

[*Class-Balanced Loss Based on Effective Number of Samples*](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500341.pdf)\
Titir Dutta, Anurag Singh, and Soma Biswas


## Dependencies:
+ Linux (tested on Ubuntu 16.04)
+ NVIDIA GPU + CUDA CuDNN
+ 7z


## Key Implementation Details:
+ [Weights for class-balanced loss](https://github.com/richardaecn/class-balanced-loss/blob/master/src/cifar_main.py#L425-L430)
+ [Focal loss](https://github.com/richardaecn/class-balanced-loss/blob/master/src/cifar_main.py#L226-L266)
+ [Assigning weights to different loss](https://github.com/richardaecn/class-balanced-loss/blob/master/src/cifar_main.py#L325-L354)

## Citation
If you find our work helpful in your research, please cite it as:
```latex
@inproceedings{dutta2020adaptive,
  title={Adaptive Margin Diversity Regularizer for Handling Data Imbalance in Zero-Shot SBIR},
  author={Dutta, Titir and Singh, Anurag and Biswas, Soma},
  booktitle={European Conference on Computer Vision},
  pages={349--364},
  year={2020},
  organization={Springer}
}
```
