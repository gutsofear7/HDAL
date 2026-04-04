# HDAL: Hierarchical Disentanglement and Depth-Aware Learning for Multi-Modal Rumor Detection

This repository contains the official implementation of our ICME 2026 paper:

> **HDAL: Hierarchical Disentanglement and Depth-Aware Learning 
> for Multi-Modal Rumor Detection**  
> Kun Lu, Hongli Zhang, Yuchen Yang, Gongzhu Yin, and Binxing Fang  
> Harbin Institute of Technology  
> IEEE International Conference on Multimedia and Expo (ICME 2026)

---

## Overview

HDAL addresses multi-modal rumor detection through three synergistic components:
- **Hierarchical Multi-modal Learning**: contrastive alignment + adversarial decoupling
- **Depth-aware Propagation Encoding**: tree transformer with temporal decay attention
- **Cross-space Fusion**: bidirectional cross-attention with adaptive gating

---

## Requirements
```bash
Python >= 3.8
PyTorch >= 1.12
transformers >= 4.20
torch-geometric >= 2.0
```



---

## Datasets

We evaluate on two benchmarks:
- **PHEME**: available at [PHEME dataset link]
- **Weibo**: available at [Weibo dataset link]

## Citation

If you find this work useful, please cite our paper:
```bibtex
@inproceedings{lu2026hdal,
  title={HDAL: Hierarchical Disentanglement and Depth-Aware Learning 
         for Multi-Modal Rumor Detection},
  author={Lu, Kun and Zhang, Hongli and Yang, Yuchen and 
          Yin, Gongzhu and Fang, Binxing},
  booktitle={IEEE International Conference on Multimedia and Expo (ICME)},
  year={2026}
}
