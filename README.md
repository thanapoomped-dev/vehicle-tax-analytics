[README.md](https://github.com/user-attachments/files/27242370/README.md)
# vehicle-tax-analytics# 🚗 Vehicle Tax Payment Behavior Analytics
### กรมการขนส่งทางบก | Tax Predictive Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optuna-green)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

> **End-to-End Machine Learning + Behavioral Economics Framework**  
> วิเคราะห์พฤติกรรมการชำระภาษีรถยนต์ ด้วย Panel Data ML และ Nudge Theory  
> เพื่อออกแบบมาตรการเชิงนโยบายที่ตรงกลุ่มเป้าหมาย

---

## 📋 สารบัญ
- [ภาพรวมโครงการ](#-ภาพรวมโครงการ)
- [โครงสร้างไฟล์](#-โครงสร้างไฟล์)
- [ขั้นตอนการวิเคราะห์](#-ขั้นตอนการวิเคราะห์-4-steps)
- [โมเดลที่ใช้](#-โมเดลที่ใช้)
- [Best Params](#-best-params-quick-run)
- [วิธีใช้งาน](#-วิธีใช้งาน)
- [ผลลัพธ์](#-ผลลัพธ์เชิงนโยบาย)
- [ข้อสังเกตด้าน Data Privacy](#-ข้อสังเกตด้าน-data-privacy)

---

## 🎯 ภาพรวมโครงการ

ภาษีรถยนต์เป็นรายได้สำคัญขององค์กรปกครองส่วนท้องถิ่น (อปท.) แต่มีรถยนต์ค้างชำระภาษีกว่า **9.9 ล้านคัน** คิดเป็นยอดค้างชำระสะสม **6,225 ล้านบาท** (ข้อมูล ณ มิถุนายน 2562)

โครงการนี้พัฒนาระบบ AI เพื่อ:
1. **พยากรณ์** ว่าใครมีแนวโน้มจะจ่ายภาษีช้าในปีถัดไป
2. **จำแนกพฤติกรรม** ออกเป็น 4 กลุ่มตามหลักเกณฑ์ ขบ.
3. **พยากรณ์** จำนวนวันที่จะล่าช้า
4. **ออกแบบมาตรการ** Nudge ที่ตรงกลุ่มเป้าหมาย

---

## 📁 โครงสร้างไฟล์

```
vehicle-tax-analytics/
│
├── 📓 Notebooks/
│   ├── Vehicle_Tax_DLT_v5.ipynb          # Development: EDA + Tuning + Full Analysis
│   └── Vehicle_Tax_PRODUCTION.ipynb      # Production: Best Params Only (Fast Run)
│
├── 📄 best_params.json                    # Optimal hyperparameters จาก Quick Run
├── 📄 requirements.txt                    # Python dependencies
├── 📄 .gitignore                          # ไม่รวมข้อมูลส่วนบุคคล
└── 📄 README.md
```

> ⚠️ **ไม่รวมไฟล์ข้อมูล** เนื่องจากเป็นข้อมูลส่วนบุคคลขั้นสูง (Sensitive Personal Data)  
> ดูรายละเอียดที่ [ข้อสังเกตด้าน Data Privacy](#-ข้อสังเกตด้าน-data-privacy)

---

## 🔬 ขั้นตอนการวิเคราะห์ (4 Steps)

### Step 1 — The Landscape: Customer Segmentation & Journey
- **Data Cleaning & Feature Engineering**: สร้าง `Days_Diff_Payment`, `Owner_Age`, `Is_Urban`, `Last_Is_Late` และ Lag Features จาก Panel Data
- **Tax Behavior Segmentation**: แบ่ง 4 กลุ่มตามหลักเกณฑ์ ขบ. ด้วย `Days_Late_Actual`
- **K-Means Clustering**: หา Optimal K ด้วย Silhouette Score
- **Executive Dashboard**: DLT Brand (ม่วง × ทอง) สำหรับนำเสนอผู้บริหาร

### Step 2 — The Core: Predictive Modeling
- **Rolling-Window Train/Test Split** แบ่งตาม `Due_Year` ≤ 2022 (Train) / > 2022 (Test)
- **Anti-Leakage**: ใช้เฉพาะ Pre-Payment Features ไม่มีข้อมูลหลังชำระ
- **3 Task หลัก**: Binary Classification · Multiclass · Regression

### Step 3 — The "Why": Explainable AI
- **SHAP Values** บน XGBoost Champion Model
- **SHAP Summary Plot** + **Dependence Plot** (Vehicle Age × Is_Urban)
- ตีความผลสำหรับผู้บริหาร

### Step 4 — The Action: Data-Driven Policy
- มาตรการ Nudge ตาม EAST Framework (Behavioural Insights Team, 2014)
- อ้างอิง Krueathep & Sitthiyot (2022): Social Norm Nudge เพิ่มอัตราชำระ +13.07%

---

## 🤖 โมเดลที่ใช้

### Binary Classification (ทำนายว่าจ่ายช้าหรือไม่)

| Model | Type | Metric |
|-------|------|--------|
| Logistic Regression (L1) | Classical | ROC-AUC |
| Random Forest | Ensemble | ROC-AUC |
| **XGBoost + Optuna** | Gradient Boosting | ROC-AUC (Champion) |
| MLP (sklearn) | Neural Network | ROC-AUC |
| CNN (1D Conv) | Deep Learning (Sequence) | ROC-AUC |
| RNN (SimpleRNN) | Deep Learning (Sequence) | ROC-AUC |
| LSTM (BiDirectional) | Deep Learning (Sequence) | ROC-AUC |

### Multiclass Classification (จำแนก 4 กลุ่มพฤติกรรม)
LogReg · Random Forest · XGBoost — Metric: **Macro F1**

### Regression (พยากรณ์วันล่าช้า)
Ridge · RF · XGBoost · ANN · CNN · RNN · LSTM — Metric: **MAE (วัน)**

---

## ⚙️ Best Params (Quick Run)

```json
{
  "XGBoost (Optuna)": {
    "n_estimators": 136,
    "max_depth": 3,
    "learning_rate": 0.04313,
    "subsample": 0.8574,
    "colsample_bytree": 0.7728,
    "min_child_weight": 12,
    "reg_lambda": 3.098
  },
  "Random Forest": {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 20
  },
  "LSTM (BiDir)": {
    "bilstm_units": 32,
    "lstm_units": 16,
    "optimizer": "Adam(lr=5e-4)",
    "epochs": 80
  }
}
```

ดูค่าครบทั้งหมดใน [`best_params.json`](best_params.json)

---

## 🚀 วิธีใช้งาน

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 2. เตรียมข้อมูล

จัดเตรียมไฟล์ CSV ของท่านให้มีโครงสร้างดังนี้:

| คอลัมน์ | ประเภท | คำอธิบาย |
|---------|--------|---------|
| `Encode_ChassisNo` | str | รหัสเข้ารหัสเลขตัวถัง |
| `Due_Date` | date | วันครบกำหนดชำระภาษี |
| `Payment_Date` | date | วันที่ชำระจริง (NaN = ยังไม่ชำระ) |
| `Due_Year` | int | ปี พ.ศ. ที่ครบกำหนด |
| `Status` | str | `จ่ายก่อน` / `จ่ายตรง` / `จ่ายช้า` |
| `Province` | str | จังหวัดที่จดทะเบียน |
| `Payment_Province` | str | จังหวัดที่ชำระ |
| `Sheet1.Owner_Birth_Date` | date | วันเกิดเจ้าของ |
| `Sheet1.Vehicle_Age_lastyear` | int | อายุรถ (ปีก่อน) |
| `Sheet1.Payment_Status` | str | สถานะปีก่อน (Lag Feature) |
| `Sheet1.Fuel_Type` | str | ประเภทเชื้อเพลิง |
| `Sheet1.Brand` | str | ยี่ห้อรถ |
| `Sheet1.Engine_CC` | float | ขนาดเครื่องยนต์ |

> ข้อมูล Sheet1.* = ข้อมูลจากปีก่อนหน้า (Panel Data Join)

### 3. รัน Development Notebook

```python
# ใน Google Colab
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. แก้ DATA_PATH ใน Cell "Load Data"
DATA_PATH = '/content/drive/My Drive/YOUR_FOLDER/your_data.csv'

# 3. ตั้ง QUICK_MODE
QUICK_MODE = True   # ทดสอบเร็ว
# QUICK_MODE = False  # Production run

# 4. Run All
```

### 4. รัน Production Notebook (ด้วย Best Params)

```python
# ใช้ Vehicle_Tax_PRODUCTION.ipynb
# เพียงแก้ DATA_PATH แล้ว Run All
# ไม่มี Optuna / ไม่มี EDA → รันเร็วกว่ามาก
```

---

## 📊 ผลลัพธ์เชิงนโยบาย

### Tax Behavior Segmentation (4 กลุ่ม ตามหลักเกณฑ์ ขบ.)

| Seg | กลุ่ม | เงื่อนไข | กลยุทธ์ Nudge |
|-----|------|---------|-------------|
| 1 | จ่ายตรงเวลา/ล่วงหน้า | Days ≤ 0 | Loyalty Program – Early Bird |
| 2 | จ่ายล่าช้า (≤1 ปี) | 0 < Days ≤ 365 | SMS + Social Norm (+13.07%) |
| 3 | จ่ายล่าช้า (1–3 ปี) | 365 < Days ≤ 1095 | จดหมายราชการ + Amnesty |
| 4 | ขาดต่อเกิน 3 ปี | Days > 1095 | Law Enforcement |

### Impact Estimation (อ้างอิงจาก Krueathep & Sitthiyot, 2022)

- ยอดภาษีค้างชำระทั่วประเทศ: **6,225 ล้านบาท**
- หาก Social Norm Nudge ตอบรับ 13.07% → เก็บภาษีเพิ่ม **296–789 ล้านบาท**
- Benefit-Cost Ratio: **2.9–3.9 เท่า**

---

## 📚 อ้างอิง

- Krueathep, W. & Sitthiyot, T. (2022). A Behavioral Approach to the Collection of Overdue Motor Vehicle Tax. *Local Administration Journal, 15*(1), 41–58.
- Behavioural Insights Team. (2014). *EAST: Four simple ways to apply behavioural insights.*
- Thaler, R. H., & Sunstein, C. R. (2008). *Nudge: Improving decisions about health, wealth, and happiness.* Penguin Books.
- Lundberg, S. & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS.*

---

## 🔒 ข้อสังเกตด้าน Data Privacy

ข้อมูลที่ใช้ในโครงการนี้เป็น **ข้อมูลส่วนบุคคลขั้นสูง** ภายใต้ พ.ร.บ. คุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562 (PDPA) ประกอบด้วย:

- ข้อมูลยานพาหนะที่เชื่อมโยงกับบุคคล
- วันเกิด และที่อยู่เจ้าของรถ
- ประวัติการชำระภาษี

**ด้วยเหตุนี้** repository นี้จึงเผยแพร่เฉพาะ **source code** เท่านั้น ไม่มีข้อมูลจริงแนบมาด้วย  
ผู้ที่ต้องการนำไปใช้ต้องจัดหาข้อมูลของตนเองและปฏิบัติตาม PDPA อย่างเคร่งครัด

---

## 🛠️ Tech Stack

```
Python 3.10+     │ Core language
Pandas / NumPy   │ Data manipulation
Scikit-learn     │ ML pipelines, classical models
XGBoost          │ Gradient boosting + Optuna tuning
TensorFlow/Keras │ CNN, RNN, LSTM, ANN
SHAP             │ Explainable AI
Optuna           │ Hyperparameter optimization
Matplotlib       │ DLT-branded visualizations (Kanit font)
Plotly           │ Interactive charts (Sankey, etc.)
SciPy            │ Chi-Square representativeness tests
Google Colab     │ Runtime environment
```

---

*โครงการนี้เป็นส่วนหนึ่งของการศึกษาระดับปริญญาโท สาขาเศรษฐศาสตร์*
