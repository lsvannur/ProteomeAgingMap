# 🧬 ProteomeAgingMap  
**The Spatiotemporal Proteome Landscape of Aging: Structural Determinants of Age-Sensitive Proteome Remodeling** Yoo, S., Vannur, L., Li, L.*, Young, C., Liu, Q., Wen, Z. T., Zhang, Y., Florens, L., Si, K., Zhuang, J., Zheng, F., & Zhou, C.#
**Single-Cell Spatial Proteomics Uncovers Molecular Interconnectivity among Hallmarks of Aging.** Yoo, S., Young, C., Li, L., Vannur, L., Zhuang, J., Zheng, F., Wu, M., Andersen, J. K., & Zhou, C.#

---

# Installation

## System Requirements

- **OS:** Linux (Ubuntu 20.04+ recommended), macOS, Windows  
- **CPU:** ≥ 8 cores recommended  
- **RAM:** ≥ 16 GB (≥ 64 GB for large-scale datasets)  
- **Storage:** ≥ 50 GB free space  

---

## 📦 Dependencies

### Software
- **Python:** 3.9 – 3.11  
- **Optional:** CUDA-enabled GPU for accelerated inference  

**Tested Environment:**
- GPU: NVIDIA RTX 6000 Ada Generation  
- CUDA: 12.9  
- NVIDIA Driver: 575.57.08  

---

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

## Inference: Predict Subcellular Localization
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
