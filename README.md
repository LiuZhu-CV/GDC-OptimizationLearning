# GDC-OptimizationLearning
  Optimization-Inspired Learning with  Architecture Augmentations and Control Mechanisms  for Low-Level Vision (IEEE TIP)

## Abstract

In recent years, there has been a growing interest in combining learnable modules with numerical optimization to solve low-level vision tasks. However, most existing approaches focus on designing specialized schemes to generate image/feature propagation. There is a lack of  the unified consideration to construct  propagative modules,  provide theoretical analyses tools and design effective learning mechanisms.
To mitigate the above issues, this paper proposes a unified optimization-inspired learning framework to aggregate Generative, Discriminative and Corrective (GDC for short) principles with strong generalization for diverse optimization models. Specifically, by introducing a general energy minimization model and formulating its descent direction from different viewpoints (\textit{i.e.,} in generative manner, based on the discriminative metric and with optimality-based correction), we construct three propagative modules to effectively solve the optimization models with flexible combinations.
We design two control mechanisms that provide the non-trivial theoretical guarantees for both fully- and partially-defined optimization formulations. Under the supporting of  theoretical guarantees, we can introduce diverse architecture augmentation strategies such as normalization and search  to ensure stable propagation with convergence and seamlessly integrate the suitable modules into the propagation respectively. Extensive experiments across varied low-level vision tasks validate the efficacy and adaptability of GDC.
  
# Citation
We thanks for the contributions of Yi He (@Heyi007), Dongxiang Shi (@sdxzy) and Dr. Shichao Cheng.
```
@article{liu2020learning,
  title={Learning optimization-inspired image propagation with control mechanisms and architecture augmentations for low-level vision},
  author={Liu, Risheng and Liu, Zhu and Mu, Pan and Lin, Zhouchen and Fan, Xin and Luo, Zhongxuan},
  journal={arXiv preprint arXiv:2012.05435},
  year={2020}
}
```
  
