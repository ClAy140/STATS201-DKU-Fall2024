# Wind Power Forecasting Using Transfer Learning

This repository contains the codebase and datasets for the research project _Wind Power Forecasting Using Transfer Learning_. The project explores the application of Long Short-Term Memory (LSTM) models, pre-trained on feature-rich datasets from China, to predict wind power generation in the United States using transfer learning. By leveraging transfer learning, the study aims to improve model accuracy and generalization across regions with diverse meteorological conditions.

---

## **1. Overview**

The project tackles the challenge of forecasting wind power generation with high accuracy across different regions. It combines meteorological data with machine learning techniques to address key research questions, including the impact of transfer learning on cross-regional model generalization. The repository includes:

- Data preprocessing and normalization scripts.
- LSTM model architecture with transfer learning capabilities.
- Scripts for training, fine-tuning, and evaluating models.
- Visualization tools for predictions and model interpretability.

---

## **2. Key Features**

- **Cross-Regional Forecasting:** Utilizes transfer learning to adapt models trained on Chinese data for U.S. regions.
- **LSTM-Based Architecture:** Implements sequential models to capture temporal dependencies in meteorological data.
- **Data Normalization:** Ensures feature compatibility across datasets from different regions.
- **Model Evaluation:** Includes RMSE, MAE, and R² metrics to assess prediction accuracy.
- **Interpretability Tools:** Generates attention heatmaps to visualize model focus on key features.

---

## **3. Datasets**

### **China Dataset**
- **Source:** Simulated plant-level wind power data.
- **Features:**
  - Wind speed (m/s)
  - Wind direction (°)
  - Air temperature (°C)
  - Atmospheric pressure (hPa)
  - Relative humidity (%)
  - Power output (MW)

### **U.S. Dataset**
- **Source:** Aggregated hourly wind power generation data across regions.
- **Features:**
  - Capacity factor proxies
  - Hourly power output

**File Access:** Filtered datasets for January 2020 (`china_jan_2020.pkl` and `us_jan_2020.pkl`) are included in the repository. Raw datasets can be accessed externally if required.

---

## **4. Codebase Structure**

| **File/Folder**       | **Description**                                                                 |
|------------------------|---------------------------------------------------------------------------------|
| `pre.py`              | Scripts for data preprocessing, normalization, and feature scaling.             |
| `main.py`             | Model training, transfer learning, and evaluation scripts.                      |
| `china_jan_2020.pkl`  | Preprocessed China dataset for January 2020.                                    |
| `us_jan_2020.pkl`     | Preprocessed U.S. dataset for January 2020.                                     |
| `README.md`           | Project documentation.                                                          |

---

## **5. Methodology**

1. **Preprocessing:**
   - China and U.S. datasets are normalized and aligned using MinMaxScaler.
   - Data is split into training (70%) and testing (30%) sets with sliding window input-output pairs.

2. **Model Training:**
   - **China Model:** LSTM trained on the Chinese dataset.
   - **Transfer Learning:** Pre-trained weights from the China model are fine-tuned on the U.S. dataset.

3. **Evaluation Metrics:**
   - **Root Mean Squared Error (RMSE):** Measures overall prediction accuracy.
   - **Mean Absolute Error (MAE):** Assesses magnitude of prediction errors.
   - **R² Score:** Evaluates model fit against actual data.

4. **Visualization:**
   - Generate attention heatmaps to interpret feature importance during predictions.
   - Plot actual vs. predicted wind power for both regions.

---

## **6. How to Run the Code**

1. **Preprocess the Data:**
   ```bash
   python pre.py
   ```
   - Filters and normalizes data from raw CSV files to produce `.pkl` files.

2. **Train Models:**
   ```bash
   python main.py
   ```
   - Trains the China model and fine-tunes the U.S. model with transferred weights.

3. **Visualize Results:**
   - Outputs prediction plots and evaluation metrics in the console.
   - Example visualization: attention heatmaps for feature analysis.

---


## **7. Future Work**

- Incorporate advanced methods like Temporal Fusion Transformers for improved temporal context modeling.
- Extend datasets to include other geographic regions for global scalability.
- Evaluate model robustness under extreme weather scenarios.

---

## **8. References**

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. _Neural Computation, 9_(8), 1735–1780.  
2. Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. _IEEE Transactions on Knowledge and Data Engineering, 22_(10), 1345–1359.  
3. Zhao, X., et al. (2022). Climate variability and renewable energy forecasting. _Renewable Energy, 195_, 1231–1243.  

