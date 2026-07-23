# ⚡ AI-Powered EV Battery Management System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen)]()

> **An Intelligent Battery Health Dashboard integrating Deep Learning for State of Health (SOH) estimation and Reinforcement Learning (DQN) for dynamic discharge optimization.**

---

## 📌 Project Overview

Electric Vehicles (EVs) are critical for sustainable transportation, but battery degradation remains a major barrier to adoption. This project addresses the challenge by building a production-ready, AI-powered Battery Management System (BMS) that:

- **Estimates** battery State of Health (SOH) with **2.03% MAPE** using a Hybrid CNN-LSTM model.
- **Optimizes** discharge strategies with a **12.6% reward improvement** using a Deep Q-Network (DQN).
- **Visualizes** real-time predictions through a Flask + Chart.js web dashboard.

---

## 🎯 Key Achievements

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **SOH Estimation MAPE** | **2.03%** | Industry Standard: 5–10% |
| **LSTM Baseline Improvement** | **22.3%** Reduction | — |
| **CNN Baseline Improvement** | **50.3%** Reduction | — |
| **RL Optimization Lift** | **+12.6%** | Project Target: +5% |
| **Data Integrity** | **Zero Leakage** | Cell-Level Train/Test Split |
| **Deployment** | **7 Artifacts** | Flask + Chart.js Dashboard |

---

## 🧠 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT: EV Telemetry                             │
│              (Voltage, Current, Temperature, SOC, Cycle)                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: SOH ESTIMATION                              │
│              Hybrid CNN-LSTM (Staged Dropout: 0.2→0.3→0.4)              │
│                    NASA Battery Ageing Dataset                          │
│                    MAPE: 2.03% | R²: 0.643                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: DISCHARGE OPTIMIZATION                      │
│              Deep Q-Network (Experience Replay + Target Network)        │
│                    Oxford Battery Degradation Dataset                   │
│                    Improvement: +12.6% vs Random Policy                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: WEB DASHBOARD                               │
│              Flask Backend → Chart.js Frontend                          │
│              CSV Upload → Real-Time Visualization                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
.
├── app.py                
├── index.html            
├── readme.md                 
├── requirements.txt           
├── models/
│   ├── cnn_model.h5         
│   ├── lstm_model.h5        
│   ├── hybrid_model.h5       
│   ├── nasa_scaler.pkl       
│   ├── q_table_baseline.pkl  
│   ├── optimized_battery_dqn.zip  
│   └── oxford_rl_scaler.pkl   
├── notebooks/                 
│   ├── nasa-pre-processing-code.ipynb
│   ├── oxford-pre-processing-code.ipynb
│   ├── soh-training-code.ipynb
│   └── rl-training-code.ipynb
├── datasets/                 
│   ├── nasa_test.csv
│   ├── nasa_train.csv
│   ├── nasa_val.csv
│   ├── oxford_train_env.parquet
│   └── oxford_test_env.parquet
└── results/           
    ├── ui-snapshot-v5.1
    ├── ui-snapshot-v5.2       
    └── ui-snapshot-v5.3
```

---

## 🛠️ Installation Guide

### System Requirements

| Component | Minimum Requirement |
|-----------|---------------------|
| **Operating System** | Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+) |
| **Python** | 3.8, 3.9, 3.10, or 3.11 (3.12 not yet fully supported by TensorFlow) |
| **RAM** | 8 GB (16 GB recommended) |
| **Storage** | 5 GB free space |
| **GPU (Optional)** | NVIDIA GPU with CUDA 11.8+ for faster training |

---

### Step 1: Install Python

**Windows:**
1. Download Python 3.10 from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **IMPORTANT:** Check the box "Add Python to PATH"
4. Click "Install Now"

**macOS:**
```bash
brew install python@3.10
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.10 python3-pip
```

**Verify Installation:**
```bash
python --version
# Should show: Python 3.10.x
```

---

### Step 2: Clone the Repository

```bash
git clone https://github.com/shiban-alam/ai-powered-ev-battery-management-system.git
cd ai-powered-ev-battery-management-system
```

---

### Step 3: Create a Virtual Environment (Recommended)

A virtual environment isolates project dependencies and prevents conflicts with other Python projects.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**You should see `(venv)` appear at the beginning of your terminal prompt.**

---

### Step 4: Install Dependencies

**Option A: Using requirements.txt (Recommended)**

Download `requirements.txt` from this repository or create it using the content below. Then run:

```bash
pip install -r requirements.txt
```

**Option B: Manual Installation (If requirements.txt is not available)**

```bash
pip install flask==2.3.3
pip install flask-cors==4.0.0
pip install tensorflow==2.15.0
pip install keras==2.15.0
pip install stable-baselines3==2.3.0
pip install gymnasium==0.29.1
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install h5py==3.10.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install tqdm==4.66.1
```

---

### Step 5: Verify Installation

Create a test file `test_install.py` with this content:

```python
import tensorflow as tf
import keras
import flask
import stable_baselines3
import numpy as np
import pandas as pd
import sklearn

print("✅ TensorFlow:", tf.__version__)
print("✅ Keras:", keras.__version__)
print("✅ Flask:", flask.__version__)
print("✅ Stable-Baselines3:", stable_baselines3.__version__)
print("✅ NumPy:", np.__version__)
print("✅ Pandas:", pd.__version__)
print("✅ Scikit-learn:", sklearn.__version__)
print("✅ All dependencies installed successfully!")
```

Run it:

```bash
python test_install.py
```

**Expected Output:**
```
✅ TensorFlow: 2.15.0
✅ Keras: 2.15.0
✅ Flask: 2.3.3
✅ Stable-Baselines3: 2.3.0
✅ NumPy: 1.24.3
✅ Pandas: 2.0.3
✅ Scikit-learn: 1.3.0
✅ All dependencies installed successfully!
```

---


### Step 6: Verify Model Files

```bash
ls -la models/
```

Expected output showing all 7 files.

---

## 🚀 How to Run Locally

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser (Chrome recommended)
- All model files present in `/models` folder

### Installation

**1. Clone the Repository**

```bash
git clone https://github.com/shiban-alam/ai-powered-ev-battery-management-system.git
cd ai-powered-ev-battery-management-system
```

**2. Install Dependencies**

```bash
pip install flask flask-cors pandas numpy tensorflow stable-baselines3 h5py scikit-learn
```

> **Note:** For GPU acceleration, install `tensorflow-gpu` instead of `tensorflow`.

**3. Start the Backend Server**

```bash
python app.py
```

Wait for the message: `✅ All Models Loaded Successfully on Local Machine!`

**4. Start the Frontend Server**

Open a second terminal:

```bash
python -m http.server 5500
```

**5. Launch the Dashboard**

Open your browser and navigate to:

```
http://localhost:5500/index.html
```

**6. Upload & Analyze**

- Upload a CSV file (e.g., `nasa_test.csv`)
- Click **"START SYSTEM"**
- Watch real-time SOH predictions and RL-guided discharge recommendations.

---

## 🧪 Common Issues & Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution:** Reinstall TensorFlow:
```bash
pip uninstall tensorflow
pip install tensorflow==2.15.0
```

### Issue 2: "numpy._core" error when loading models

**Solution:** This is a NumPy 2.0 compatibility issue. Downgrade NumPy:
```bash
pip install numpy==1.24.3 --force-reinstall
```

### Issue 3: Flask server not starting

**Solution:** Check if port 5000 is in use:
```bash
# Windows
netstat -ano | findstr :5000

# Mac/Linux
lsof -i :5000
```

Change the port in `app.py`:
```python
app.run(host='127.0.0.1', port=5001)
```

### Issue 4: CORS errors in browser

**Solution:** Ensure `flask-cors` is installed:
```bash
pip install flask-cors
```

### Issue 5: Model files not found

**Solution:** Check file paths in `app.py`. The models should be in a folder named `models/` in the same directory as `app.py`.

### Issue 6: Out of Memory (OOM) errors

**Solution:** Reduce batch size or use CPU-only mode:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

---

## 📊 Model Performance Summary

### Phase 1: SOH Estimation (NASA Dataset)

| Model | MAPE | MAE | RMSE | R² Score |
|-------|------|-----|------|----------|
| CNN (Spatial Baseline) | 4.096% | 0.0293 | 0.0348 | -0.281 |
| LSTM (Temporal Baseline) | 2.621% | 0.0187 | 0.0222 | 0.477 |
| **Hybrid CNN-LSTM (Proposed)** | **2.037%** | **0.0144** | **0.0184** | **0.643** |

### Phase 2: Discharge Optimization (Oxford Dataset)

| Algorithm | Improvement vs Random | Training Time | Scalability |
|-----------|-----------------------|---------------|-------------|
| Q-Learning (Tabular) | +5.8% | ~5 min (CPU) | Limited (2,184 states) |
| **DQN (Proposed)** | **+12.6%** | **~20 min (GPU)** | **High (Continuous States)** |

---

## 🎓 Research Contributions

This project addresses three critical gaps in existing battery management research:

1. **Zero-Leakage Validation:** Strict cell-level train/test splits ensure reported metrics reflect generalization to physically unseen batteries — unlike many papers that leak data across cycles of the same cell.

2. **Hybrid Architecture:** The CNN-LSTM combination extracts spatial features before temporal reasoning, achieving 22.3% improvement over LSTM alone.

3. **Production Deployment:** Unlike theoretical models, this system is deployed via Flask + Chart.js, making AI-powered BMS accessible without client-side installation.

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| **Deep Learning** | TensorFlow 2.x, Keras |
| **Reinforcement Learning** | Stable-Baselines3 (DQN), PyTorch 2.10.0 |
| **Data Processing** | NumPy, Pandas, Scikit-learn |
| **Web Framework** | Flask 2.x (Backend), Chart.js (Frontend) |
| **Development** | Google Colab (GPU Training), Jupyter Notebook, Git/GitHub |

---

## 🧪 Datasets Used

| Dataset | Purpose | Cells | Records |
|---------|---------|-------|---------|
| **NASA Battery Ageing Dataset** | SOH Estimation (Phase 1) | B0005, B0006, B0007, B0018 | 24,000+ cycles |
| **Oxford Battery Degradation Dataset** | RL Discharge Optimization (Phase 2) | Cell1–Cell8 | 7,669,585 timesteps |

---

## 🔮 Future Work

- **Multi-Chemistry Generalization:** Fine-tune the Hybrid CNN-LSTM on LFP and NMC cell datasets.
- **Hardware-in-the-Loop (HiL) Testing:** Deploy on Raspberry Pi / NVIDIA Jetson.
- **Federated Learning:** Train models across EV fleets without sharing raw telemetry.
- **Explainability:** Integrate SHAP or LIME for feature-level insights.
- **Mobile Application:** Android/iOS companion app for real-time monitoring.

---

## 👥 Team

| Name | Roll No. | Role |
|------|----------|------|
| **Md. Shiban Alam** | 12230622057 | Data Preprocessing, Model Training, Frontend Dashboard, Flask Backend |
| **Syed Sarafeena Ali** | 12230622028 | Data Preprocessing, Literature Survey, Documentation |
| **Purbasha Roy** | 12230622007 | Literature Survey, UI Feedback |
| **Oindrila Sain** | 12230622005 | Literature Survey, Documentation |

**Under the Guidance of:** Mrs. Amrita Bhattacharya (Assistant Professor, Dept. of AIML)

**Head of Department:** Mr. Amit Kumar Siromoni

---

## 📄 License

This project is submitted in partial fulfillment of the requirements for the degree of **B.Tech in Artificial Intelligence & Machine Learning** at **St. Thomas' College of Engineering and Technology**, affiliated to **Maulana Abul Kalam Azad University of Technology, West Bengal**.

© 2026 — All Rights Reserved.

---

## 🙏 Acknowledgements

- **NASA Prognostics Center of Excellence** - for the Battery Ageing Dataset
- **Oxford Battery Intelligence Lab** - for the Battery Degradation Dataset
- **TensorFlow & Keras Teams** - for deep learning frameworks
- **Stable-Baselines3 Team** - for reinforcement learning implementations


---

## 📧 Contact

For any queries regarding this project, please reach out to:

- **Md. Shiban Alam** — [mdshibanalam@gmail.com]

---

**⭐ If you found this project useful, please consider giving it a star on GitHub!**