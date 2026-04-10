# ProteomeAgingMap  
1. [**The Spatiotemporal Proteome Landscape of Aging: Structural Determinants of Age-Sensitive Proteome Remodeling**](https://www.biorxiv.org/content/10.64898/2026.02.26.708310v1) Yoo, S., Vannur, L., Li, L.*, Young, C., Liu, Q., Wen, Z. T., Zhang, Y., Florens, L., Si, K., Zhuang, J., Zheng, F., & Zhou, C.#

2. [**Single-Cell Spatial Proteomics Uncovers Molecular Interconnectivity among Hallmarks of Aging.**](https://www.biorxiv.org/content/10.64898/2026.02.26.708335v1) Yoo, S., Young, C., Li, L., Vannur, L., Zhuang, J., Zheng, F., Wu, M., Andersen, J. K., & Zhou, C.#

---

## Overview

**ProteomeAgingMap** provides a computational pipeline for large-scale analysis of proteome remodeling during aging. The framework enables:

- Preprocessing of microscopy `.nd2` datasets  
- Cell segmentation and postprocessing  
- Bud scar detection and quantification  
- GFP intensity extraction  
- Subcellular localization prediction using DeepLoc (2D) and a ResNet 3D model  
- Ensemble-based localization inference  

---

# Installation

## System Requirements and Dependencies

- **Operating System:** Linux (Ubuntu 20.04+ recommended), macOS, Windows  
- **Python:** 3.9 – 3.11  
- **CPU:** ≥ 8 cores recommended  
- **RAM:** ≥ 16 GB (≥ 64 GB recommended for large-scale datasets)  
- **Storage:** ≥ 50 GB free space  
- **GPU (optional):** CUDA-enabled GPU for accelerated inference  

**Tested Environment:**
- GPU: NVIDIA RTX 6000 Ada Generation  
- CUDA: 12.9  
- NVIDIA Driver: 575.57.08  

## Setup Instructions

Clone the repository and set up the conda environment:

```bash
git clone https://github.com/lsvannur/ProteomeAgingMap.git
cd ProteomeAgingMap/screening_codes
conda env create -f environment.yml
conda activate gpuenv
mkdir input_batches
```

## Edit config.yaml
### Parameter Description
```yaml
image_folder: Path to folder containing .nd2 files
model_budscar_path: Path to trained 3D model
screen_folder: Path to screening_codes directory
local_batch_folder_path: Create this folder (for batching inputs)
output_batch_folder_path: Folder where .h5 outputs will be saved
CPU_COUNT: Number of CPU cores
BATCH_SIZE: Adjust based on RAM/GPU
```

``` bash
python main.py
```

## Inference: To Predict Subcellular Localization over those cells run
```bash
cd deeploc_3D_ensemble_codes
```

Download the pretrained models:

### edit infer_config_h5.yaml
```yaml
running_batch_name: 'plate_name'
batch_folder_path: '../input_batches/'
modelpath_3d: '../../models/3d-Resnet10_3D_24_0.94.pth'
modelpath_2d: '../../models/transfer_v2/full_training_dataset.ckpt' 
```

### run inference:
``` bash 
python infer_3d_h5.py
