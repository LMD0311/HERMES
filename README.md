<div  align="center">    
 <img src="./figures/logo.jpg" width = "150"  align=center />
</div>

</div>

<div align="center">
<h3>HERMES: A Unified Self-Driving World Model for Simultaneous <br>3D Scene Understanding and Generation</h3>




[Xin Zhou](https://lmd0311.github.io/)<sup>1\*</sup>, [Dingkang Liang](https://dk-liang.github.io/)<sup>1\*†</sup>, Sifan Tu<sup>1</sup>, [Xiwu Chen](https://scholar.google.com/citations?user=PVMQa-IAAAAJ&hl=en)<sup>3</sup>, [Yikang Ding](https://scholar.google.com/citations?user=gdP9StQAAAAJ&hl=en)<sup>2†</sup>, Dingyuan Zhang<sup>1</sup>, Feiyang Tan<sup>3</sup>,<br> [Hengshuang Zhao](https://scholar.google.com/citations?user=4uE10I0AAAAJ&hl=en)<sup>4</sup>, [Xiang Bai](https://scholar.google.com/citations?user=UeltiQ4AAAAJ&hl=en)<sup>1</sup>

<sup>1</sup>  Huazhong University of Science & Technology, <sup>2</sup>  MEGVII Technology, <br><sup>3</sup>  Mach Drive, <sup>4</sup>  The University of Hong Kong

(\*) Equal contribution. (†) Project leader.

<!-- [![arXiv](https://img.shields.io/badge/Arxiv-2402.10739-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2402.10739) -->
[![Project](https://img.shields.io/badge/Homepage-project-orange.svg?logo=googlehome)](https://lmd0311.github.io/HERMES/)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLMD0311%2FHERMES&count_bg=%23417A15&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)

Check our *awesome* for the latest World Models! [![Awesome World Model](https://img.shields.io/badge/GitHub-awesome_world_model-blue?logo=github)](https://github.com/LMD0311/Awesome-World-Model)
![Stars](https://img.shields.io/github/stars/LMD0311/Awesome-World-Model)



</div>

## 📣 News

- **[2025.01.22]** Release the demo. Check it out and give it a star 🌟!

- **[2025.01.22]** Release the [paper](xxx).

 <div  align="center">    
 <img src="./figures/intro.png" width = "888"  align=center />
</div>

## Abstract

Driving World Models (DWMs) have become essential for autonomous driving by enabling future scene prediction. However, existing DWMs are limited to scene generation and fail to incorporate scene understanding, which involves interpreting and reasoning about the driving environment. In this paper, we present a unified Driving World Model named **HERMES**<sup>1</sup>. Through a unified framework, we seamlessly integrate scene understanding and future scene evolution (generation) in driving scenarios. Specifically, **HERMES** leverages a Bird‘s-Eye View (BEV) representation to consolidate multi-view spatial information while preserving geometric relationships and interactions. Additionally, we introduce world queries, which incorporate world knowledge into BEV features via causal attention in the Large Language Model (LLM), enabling contextual enrichment for both understanding and generation tasks. We conduct comprehensive studies on nuScenes and OmniDrive-nuScenes datasets to validate the effectiveness of our method. **HERMES** achieves state-of-the-art performance, reducing generation error by 32.4% and improving understanding metrics such as CIDEr by 8.0%.



> 1. In Greek mythology, [Hermes](https://en.wikipedia.org/wiki/Hermes) serves as the messenger of the gods. Similarly, this paper proposes a simple yet effective framework that unifies understanding and generation as a driving world model, facilitating knowledge transfer across tasks. The logo inspired by Hermes’ shoes.

## Overview

<div  align="center">    
 <img src="./figures/pipeline.jpg" width = "888"  align=center />
</div>



## Demo

<div  align="center">    
 <img src="./figures/scene1.gif" width = "999"  align=center />
 <center> Example 1 </center> <br>
</div>

<div  align="center">    
 <img src="./figures/scene2.gif" width = "999"  align=center />
 <center> Example 2 </center> <br>
</div>

<div  align="center">    
 <img src="./figures/scene3.gif" width = "999"  align=center />
 <center> Example 3 </center> <br>
</div>


## Main Results

<div  align="center">    
 <img src="./figures/main_results.png" width = "888"  align=center />
</div>


## Getting Started

Coming soon. 

## To Do

- [x] Release demo.
- [ ] Release checkpoints.
- [ ] Release training code.

## Acknowledgement

This project is based on BEVFormer v2 ([paper](https://arxiv.org/abs/2211.10439), [code](https://github.com/fundamentalvision/BEVFormer)), InternVL ([paper](https://arxiv.org/abs/2404.16821), [code](https://github.com/OpenGVLab/InternVL)), UniPAD ([paper](https://arxiv.org/abs/2310.08370), [code](https://github.com/Nightmare-n/UniPAD)), OminiDrive ([paper](https://arxiv.org/abs/2405.01533), [code](https://github.com/NVlabs/OmniDrive)). Thanks for their wonderful works.

## Citation

If you find this repository useful in your research, please consider giving a star ⭐ and a citation.
```bibtex
@article{zhou2025hermes,
  title={HERMES: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation},
  author={Zhou, Xin and Liang, Dingkang and Tu, Sifan and Chen, Xiwu and Ding, Yikang and Zhang, Dingyuan and Tan, Feiyang and Zhao, Hengshuang and Bai, Xiang},
  journal={arXiv preprint arXiv:},
  year={2025}
}
```
