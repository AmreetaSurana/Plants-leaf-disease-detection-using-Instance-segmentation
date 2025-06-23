
# 🌿 Plants Leaf Disease Detection using Instance Segmentation

This project implements an **Instance Segmentation model** to detect individual plant leaves and accurately identify diseased areas in real-world conditions using **deep learning and clustering-based post-processing.**

The project is based on the spatial embedding and clustering methodology proposed by Neven et al., adapted for leaf-level disease detection.

---

## 📂 Project Structure


├── dataset-mini/         # Sample dataset (for quick evaluation)
├── logs/                 # Training logs and visualizations
├── src/                  # Source code for training and reporting
│   ├── train.py          # Model training
│   ├── report.py         # Post-processing and clustering
│   ├── exp/              # Pretrained model and checkpoints
├── train\_config.py       # Training configuration
├── report\_config.py      # Clustering and reporting configuration
└── README.md             # Project documentation

````

---

## 🚀 Features
- ✅ **Instance-level segmentation**: Separates individual leaves for precise disease detection.
- ✅ **Post-processing with clustering**: Groups embeddings into distinct leaf instances.
- ✅ **Custom dataset training support**: Easily adaptable to your own plant leaf datasets.
- ✅ **Pretrained model available**: Quick-start evaluation using included model checkpoints.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/AmreetaSurana/Plants-leaf-disease-detection-using-Instance-segmentation.git
cd Plants-leaf-disease-detection-using-Instance-segmentation
````

### 2. Create the Conda Environment

```bash
conda create -n plant-segmentation python=3.7
conda activate plant-segmentation

# Install core packages
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
conda install matplotlib tqdm scikit-image pandas
conda install -c conda-forge tensorboard opencv pycocotools
conda install -c anaconda future h5py
```

---

## 📊 Training & Inference

### Train on Your Own Dataset

```bash
export DATASET_DIR=/path/to/your/dataset

# Adjust parameters in train_config.py as needed
python src/train.py
python src/report.py
```

### Run Pretrained Model on Sample Dataset

```bash
export DATASET_DIR=./dataset-mini

# In train_config.py:
# Set only_eval = True
# Provide resume_path to pretrained model
python src/train.py
python src/report.py
```

---


## 🧩 Configuration Files

* `train_config.py`: Dataset path, training parameters, evaluation settings.
* `report_config.py`: Post-processing parameters including clustering thresholds.

---

## 📌 Key Notes

* The project uses **instance embedding-based segmentation** with clustering for precise leaf and disease boundary detection.
* Requires **CUDA 9.0** and **PyTorch 1.1.0** as per current setup.
* License: **CC BY-NC** (Non-commercial use only)

---

## 🔮 Future Improvements

* ✅ Upgrade to newer versions of PyTorch and CUDA.
* ✅ Expand dataset coverage across more plant species and disease types.
* ✅ Develop real-time or mobile-friendly inference pipeline.
* ✅ Integrate advanced visualization dashboards.

---

## 🤝 Acknowledgements

This work is inspired by the method presented by Neven et al.
[Original Paper Repository](https://github.com/nautran/instance-segmentation-pytorch)

---

## 📬 Contact

For queries or collaboration, please reach out to:
**Amreeta Surana**
[GitHub Profile](https://github.com/AmreetaSurana)

---
