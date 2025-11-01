# 🌊 Coastal Erosion Prediction using Neural Networks  
### IEEE GRSS Coastal Erosion Monitoring Project

---

## 📘 Overview
This project is part of the **IEEE GRSS Coastal Erosion Monitoring Initiative**, aiming to develop a predictive model for **coastal erosion rate estimation** using geographical, demographic, and environmental data.  

The model uses a **PyTorch-based neural network** trained on a dataset containing:
- 🌍 Country & Continent  
- 📆 Year  
- 👥 Population  
- 🏞️ Area, Latitude, Longitude, Elevation  
and predicts **coastal erosion rate** values.

---

## ⚙️ Features
- ✅ End-to-end training pipeline (from preprocessing to evaluation)  
- 🔢 Automated scaling and one-hot encoding using `scikit-learn`  
- 🧠 Deep Neural Network implemented in **PyTorch**  
- 📊 Model evaluation with MSE, RMSE, MAE, and R² metrics  
- 💾 Preprocessing pipeline and model weights saved for deployment  

---

<details>
<summary>🧩 <b>Project Workflow</b> (click to expand)</summary>

### 1. **Data Loading**
Reads the Excel file (`cedt.xlsx`) and assigns column headers:


---

### 2. **Data Preprocessing**
- **Numerical columns** → Standardized using `StandardScaler`  
- **Categorical columns** (`country`, `continent`) → One-hot encoded  
- Combined via `ColumnTransformer`  
- Converted into **PyTorch tensors**

---

### 3. **Neural Network Architecture**
| Layer | Input | Output | Activation |
|--------|--------|---------|-------------|
| Linear | input_dim | 64 | ReLU |
| Linear | 64 | 32 | ReLU |
| Linear | 32 | 1 | — |

**Loss:** MSE  
**Optimizer:** Adam (lr = 0.001)  
**Epochs:** 100  

---

### 4. **Training**
- 80% data used for training, 20% for testing  
- Batch size: 32  
- Loss printed every 10 epochs  

---

### 5. **Evaluation Metrics**
After training, model performance is evaluated using:
- **MSE (Mean Squared Error)**  
- **RMSE (Root Mean Squared Error)**  
- **MAE (Mean Absolute Error)**  
- **R² Score**

---

### 6. **Model Saving**
After successful training:


</details>

---

## 🧠 Example Output
✅ Data loaded successfully.
Epoch [10/100] - Loss: 0.0532
...
✅ Test MSE: 0.0471

📊 Evaluation Metrics:
• MSE (Mean Squared Error): 0.0471<br>
• RMSE (Root Mean Squared Error): 0.2171<br>
• MAE (Mean Absolute Error): 0.1524<br>
• R² Score (Model Accuracy): 0.8932<br>
💾 Model weights saved as 'erosion_model_weights.pth'<br>
💾 Preprocessor saved as 'erosion_preprocessor.pkl'<br>

---

## 📦 Requirements

### 🐍 Python version
`Python 3.9+` recommended

### 📦 Dependencies
Install required packages:
```bash
pip install pandas torch scikit-learn joblib openpyxl numpy
```
project directory:
```
│
├── cedt.xlsx                    # Input dataset (coastal erosion data)
├── erosion_model.py             # Training & evaluation script
├── erosion_model_weights.pth    # Saved PyTorch model weights
├── erosion_preprocessor.pkl     # Saved preprocessing pipeline
└── README.md                    # Documentation file
🚀 How to Run
```
Ensure your dataset file cedt.xlsx is in the project directory.

Run the model script:

python erosion_model.py


The script will:

Train the neural network
Display evaluation metrics
Save model and preprocessing artifacts

### 🧭 Future Scope

🛰️ Integrate satellite imagery using CNN models

⏱️ Add temporal erosion tracking (multi-year prediction)

🌐 Deploy model via Flask, Django, or FastAPI

🗺️ Visualize erosion heatmaps using GIS tools

## 👨‍💻 Authors

IEEE GRSS Coastal Monitoring Team
Developed by: Sannihith Reddy M
Institution: MAHE Bengaluru

Department: MIT CSE

