# ETL Pipeline - mk4

This project is an **AI-driven ETL (Extract, Transform, Load) pipeline** with a **Flask-based dashboard** for data processing, analysis, and visualization. It integrates **machine learning for outlier detection and clustering**, stores processed data in **MongoDB**, and offers an interactive web interface for insights.

---

## **Key Features**  

### **1. ETL Pipeline**  
- **Data Cleaning**: Handles missing values, encodes categorical features, and scales numerical features.
- **Outlier Detection**: Uses **Isolation Forest** to detect and remove anomalies.
- **Clustering**: Applies **K-Means** to group similar data points.
- **Hyperparameter Optimization**: Uses **Optuna** to fine-tune clustering and anomaly detection.
- **Data Storage**: Saves processed data as **CSV, Parquet**, and uploads it to **MongoDB Atlas**.

### **2. Dashboard**  
- **Cluster Analysis**: Visualizes cluster distributions and characteristics.
- **Feature Distributions**: Histograms, box plots, and KDE plots.
- **Correlation Analysis**: Heatmaps for feature relationships.
- **Statistical Summary**: Descriptive statistics, skewness, and kurtosis.
- **Data Download**: Allows downloading processed data.

### **3. Deployment & Automation**  
- **Flask API**: Serves the dashboard via a web interface.
- **Docker Support**: Deploy as a containerized application.
- **MongoDB Integration**: Stores processed data in a NoSQL database.

---

## **Installation & Setup**  

### **1️⃣ Prerequisites**  
- **Python 3.8+** installed
- Install required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- **Ngrok** for exposing the local server publicly:
```bash
pip install pyngrok
```
- **MongoDB Atlas** (or local MongoDB instance for data storage)

### **2️⃣ Running the Project Locally**  
```bash
python etl_dashboard.py
```
- Open the dashboard at: **http://127.0.0.1:5001/**  

### **3️⃣ Using Docker (Optional)**  
#### **Build and Run the Docker Container:**  
```bash
docker build -t etl-dashboard .
docker run -p 5001:5001 etl-dashboard
```
### **4️⃣ Using Ngrok for Public Access**

If you want to access the dashboard from anywhere, use Ngrok:
```bash
ngrok http 5001
```
Copy the Ngrok public URL and open it in your browser.

#### **With Docker Compose (If Database Integration is Required)**  
```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "5001:5001"
    depends_on:
      - db
  db:
    image: mongo
    ports:
      - "27017:27017"
```
Run with:
```bash
docker-compose up --build
```


---

## **Troubleshooting**  

| Issue | Solution |
|--------|---------|
| Flask app not starting | Ensure dependencies are installed (`pip install -r requirements.txt`). |
| MongoDB connection failed | Verify MongoDB URI and check network access. |
| Docker permission error | Run commands with `sudo` on Linux/Mac. |
| Dashboard not loading | Ensure Flask server is running on port `5001`. |

---

## **Dependencies**  

- **Core Libraries**: Flask, Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib
- **Database**: PyMongo (for MongoDB)
- **Optimization**: Optuna
- **Storage**: PyArrow, FastParquet
- **Containerization**: Docker

---

## **License**  
This project is open-source and available under the **MIT License**.

---

### **Author & Contributions**  
Feel free to contribute! Fork the repo, create a branch, and submit a pull request. For inquiries, contact rishipkulkarni@gmail.com
