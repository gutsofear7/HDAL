# HDAL: Hierarchical Disentanglement and Depth-Aware Learning

Official implementation of "HDAL: Hierarchical Disentanglement and Depth-Aware Learning for Multi-Modal Rumor Detection" (ICME 2026).

## 🎯 Overview

HDAL addresses multi-modal rumor detection through hierarchical learning across three synergistic components:

1. **Hierarchical Multi-modal Learning**: Disentangles text-image features via contrastive alignment and adversarial decoupling to expose fine-grained conflicts
2. **Depth-aware Propagation Encoding**: Captures credibility evolution across cascade structure through temporal decay attention and structural masking
3. **Cross-space Fusion**: Projects heterogeneous representations into unified space for adaptive evidence integration

## 📊 Results

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| PHEME   | 91.29%   | 90.04%    | 88.87% | 89.41%   |
| Weibo   | 92.42%   | 91.10%    | 92.33% | 92.21%   |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/HDAL.git
cd HDAL

# Install dependencies
pip install -r requirements.txt
```

### PHEME Dataset

```bash
# 1. Prepare data (download PHEME dataset and organize according to data/README.md)

# 2. Extract embeddings
cd pheme/preprocessing
python emb.py

# 3. Train model
cd ../model
python model.py

# 4. Run experiments (ablation studies and visualization)
cd ../experiments
python run_experiments.py --run_ablation --run_visualization
```

### Weibo Dataset

```bash
# 1. Prepare data (contact authors for Weibo dataset access)

# 2. Extract text embeddings
cd weibo/preprocessing
python text_emb.py

# 3. Extract image embeddings
python image_emb.py

# 4. Train model
cd ../model
python model.py
```

## 📁 Project Structure

```
HDAL/
├── pheme/                      # PHEME dataset implementation
│   ├── preprocessing/          # Data preprocessing and embedding extraction
│   │   └── emb.py             # Extract text, image, and tweet embeddings
│   ├── model/                 # Model implementation
│   │   └── model.py           # Complete HDAL model
│   └── experiments/           # Experimental scripts
│       └── run_experiments.py # Ablation studies and visualization
│
├── weibo/                     # Weibo dataset implementation
│   ├── preprocessing/         # Data preprocessing
│   │   ├── text_emb.py       # Text embedding extraction
│   │   └── image_emb.py      # Image embedding extraction (CLIP)
│   └── model/                # Model implementation
│       └── model.py          # Complete HDAL model
│
├── data/                      # Data format documentation
│   └── README.md             # Dataset structure and format
│
└── docs/                      # Additional documentation
    └── ANALYSIS.md           # Complete analysis report
```

## 📖 Documentation

- [Data Format Guide](data/README.md) - Dataset structure and format specifications
- [Complete Analysis Report](docs/ANALYSIS.md) - Detailed implementation analysis

## 🔗 Datasets

- **PHEME**: [Download from Figshare](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)
- **Weibo**: Contact authors for access (lukun@hit.edu.cn)

## 🛠️ Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- transformers >= 4.51.0 (for PHEME with Qwen3 support)
- See [requirements.txt](requirements.txt) for complete list

## 🔬 Key Features

### Hierarchical Multi-modal Learning
- **Contrastive Alignment**: Establishes cross-modal alignment at shared semantic levels
- **Adversarial Disentanglement**: Separates modality-invariant and modality-specific features

### Depth-aware Propagation Learning
- **Tree Transformer**: Models propagation as directed trees with temporal decay
- **Structural Masking**: Enforces hierarchical dependencies in attention mechanism

### Cross-space Fusion
- **Bidirectional Cross-attention**: Enables interaction between semantic and propagation spaces
- **Dynamic Gating**: Performs adaptive evidence integration based on learned reliability patterns

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{hdal2026,
  title={HDAL: Hierarchical Disentanglement and Depth-Aware Learning for Multi-Modal Rumor Detection},
  author={Lu, Kun and Zhang, Hongli and Yang, Yuchen and Yin, Gongzhu and Fang, Binxing},
  booktitle={IEEE International Conference on Multimedia and Expo (ICME)},
  year={2026}
}
```

## 📧 Contact

- **Kun Lu**: lukun@hit.edu.cn
- **Issues**: Please use [GitHub Issues](https://github.com/Gutsofear7/HDAL/) for bug reports and feature requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work was supported by:
- Natural Science Foundation of Heilongjiang Province of China (Grant No. LH2023F018)
- Fundamental Research Funds for the Central Universities (Grant No. LH2023F018)
- Key Research and Development Program of Heilongjiang Province of China (Grant No. JD2023SJ07)
