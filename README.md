# ETL-ML
This is an **AI-driven ETL (Extract, Transform, Load) pipeline** that automates the data processing workflow with machine learning:

---

## **1. Library Installations & Imports**  
- Installs required libraries like `pandas`, `scikit-learn`, `autogluon`, `optuna`, etc.  
- Imports modules for **data handling (`pandas`, `numpy`, `dask`)**, **machine learning (`KNNImputer`, `IsolationForest`, `KMeans`)**, **AutoML (`autogluon`)**, **visualization (`matplotlib`, `seaborn`)**, and **database (`sqlite3`)**.

---

## **2. AI Configuration (config.yaml)**  
- Stores **ETL settings** in a YAML file for flexibility.  
- Contains:  
  - **Imputation Strategy:** Uses K-Nearest Neighbors (KNN) with `n_neighbors=5` for filling missing values.  
  - **Outlier Detection:** Uses `IsolationForest` with `contamination=0.05` (5% data considered outliers).  
  - **Clustering:** Uses `KMeans` with `n_clusters=4` for passenger segmentation.  
  - **AutoML:** Uses `autogluon` with a `120s` time limit and `best_quality` presets.

---

## **3. AI Logger (AILogger Class)**
- A logging system that records ETL execution steps in a log file (`ai_etl.log`).  
- Provides **INFO** and **WARNING** logging levels.

---

## **4. Data Extraction (SmartDataExtractor Class)**
- Extracts data from **CSV, JSON, or SQL databases** based on user input.  
- Reads the file and converts it into a **pandas DataFrame**.  
- Logs extraction success or failure.

---

## **5. Data Transformation (AIDataProcessor Class)**
- **Cleans Data:**  
  - Detects **missing values** and fills them using **KNN Imputation**.  
  - Detects **outliers** using **Isolation Forest** and removes them.  

- **Enhances Data:**  
  - Uses **KMeans clustering** on `Age` and `Fare` columns to segment passengers.  
  - If these columns are missing, it logs a warning and skips clustering.

---

## **6. Data Loading**
- Stores the processed DataFrame into an **SQLite database (`titanic_ai.db`)** in a table called **`titanic_pro`**.

---

## **7. Visualization**
- **Visualizes insights using Seaborn & Matplotlib:**  
  - Age distribution  
  - Fare vs. Survival rate  
  - Passenger segmentation (if available)

---

## **8. Execution Pipeline**
- **Extracts** Titanic dataset from CSV (`/content/titanic.csv`).  
- **Transforms** data (cleans, imputes, removes outliers, clusters).  
- **Loads** processed data into SQLite database.  
- **Generates** an AI dashboard with insights.  
- If extraction fails, the pipeline halts.

---

## **Final Summary**
- **AI-driven ETL pipeline** for handling missing data, outliers, and clustering.  
- Uses **AutoML (`autogluon`)** for further predictions.  
- Saves cleaned data into an **SQLite database**.  
- Generates **AI-powered dashboards** for visualization.  
- Implements **logging and YAML-based configuration** for flexibility.  

