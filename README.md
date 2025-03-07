### **Documentation for the AI-Driven ETL Pipeline Dashboard**

This project is an **AI-driven ETL (Extract, Transform, Load) pipeline** with a **Flask-based dashboard** for visualizing and analyzing processed data. It includes data cleaning, outlier detection, clustering, and statistical analysis, all presented in an interactive web interface.

---

### **Key Features**

1. **Data Processing**:
   - **Data Cleaning**: Handles missing values, encodes categorical features, and scales numerical features.
   - **Outlier Detection**: Uses Isolation Forest to detect and remove outliers.
   - **Clustering**: Applies K-Means clustering to group similar data points.

2. **Dashboard**:
   - **Cluster Analysis**: Visualizes cluster distributions and characteristics.
   - **Feature Distributions**: Displays histograms and box plots for numerical features.
   - **Correlation Analysis**: Shows feature correlation matrices.
   - **Statistical Analysis**: Provides descriptive statistics, skewness, and kurtosis.
   - **Data Download**: Allows downloading processed data in CSV format.

3. **Automation**:
   - **Hyperparameter Optimization**: Uses Optuna to automatically find the best parameters for outlier detection and clustering.
   - **Pipeline Persistence**: Saves the preprocessing pipeline and processed data for reuse.

4. **Visualization**:
   - Interactive plots using Seaborn and Matplotlib.
   - 3D scatter plots for cluster visualization.
   - Heatmaps for correlation analysis.

5. **Error Handling**:
   - Graceful error pages with detailed messages.
   - Recovery options to return to the dashboard.

---

### **Steps to Use in Google Colab**

1. **Run the Code**:
   - Copy the entire code into a Colab notebook.
   - Click the play button to execute the code.

2. **Access the Dashboard**:
   - After running the code, a public ngrok URL will be printed.
   - Open the URL in your browser to access the dashboard.

3. **Explore the Dashboard**:
   - Use the navigation bar to switch between different analysis pages.
   - Interact with visualizations and download processed data.

4. **Stop the Server**:
   - To stop the server, interrupt the Colab runtime (Runtime → Interrupt Execution).

---

### **Steps to Use Locally**

1. **Prerequisites**:
   - Install Python 3.8+.
   - Install the required libraries:
     ```bash
     pip install flask pandas numpy scikit-learn seaborn matplotlib optuna pyngrok joblib pyarrow
     ```

2. **Download the Code**:
   - Save the code to a file, e.g., `etl_dashboard.py`.

3. **Run the Code**:
   - Open a terminal and navigate to the directory containing the script.
   - Run the script:
     ```bash
     python etl_dashboard.py
     ```

4. **Access the Dashboard**:
   - The script will print a local URL (e.g., `http://127.0.0.1:5001`).
   - Open the URL in your browser to access the dashboard.

5. **Stop the Server**:
   - Press `Ctrl+C` in the terminal to stop the server.

---

### **Code Structure**

1. **Configuration**:
   - `Config` class contains all configuration parameters (e.g., data URL, features, hyperparameter ranges).

2. **ETL Pipeline**:
   - `ProductionETL` class handles data processing, including cleaning, outlier detection, and clustering.

3. **Dashboard**:
   - Flask app with routes for different analysis pages.
   - Interactive visualizations using Seaborn and Matplotlib.

4. **Error Handling**:
   - Custom error pages with recovery options.

5. **Main Execution**:
   - Sets up the environment, runs the pipeline, and starts the Flask server.

---

### **Customization**

1. **Data Source**:
   - Update `Config.DATA_URL` to use a different dataset.

2. **Features**:
   - Modify `Config.NUMERICAL_FEATURES` and `Config.CATEGORICAL_FEATURES` to match your dataset.

3. **Hyperparameters**:
   - Adjust `Config.CONTAMINATION_RANGE` and `Config.N_CLUSTERS_RANGE` for outlier detection and clustering.

4. **Visualizations**:
   - Customize plots by modifying the Seaborn and Matplotlib code in the respective routes.

---

### **Example Workflow**

1. **Load Data**:
   - The pipeline loads data from the specified URL.

2. **Clean Data**:
   - Missing values are imputed, and categorical features are encoded.

3. **Optimize Parameters**:
   - Optuna finds the best parameters for outlier detection and clustering.

4. **Process Data**:
   - Outliers are removed, and data is clustered.

5. **Visualize Results**:
   - The dashboard displays interactive visualizations and statistics.

6. **Download Data**:
   - Processed data can be downloaded in CSV format.

---

### **Troubleshooting**

1. **Ngrok Issues**:
   - Ensure your ngrok token is valid.
   - Restart the server if the ngrok URL becomes inaccessible.

2. **Data Errors**:
   - Check for missing or invalid values in the dataset.
   - Update the feature lists in the `Config` class if necessary.

3. **Visualization Errors**:
   - Ensure all required columns are present in the dataset.
   - Check for numeric data types in feature columns.

---

### **Dependencies**

- **Python Libraries**:
  - Flask
  - Pandas
  - NumPy
  - Scikit-learn
  - Seaborn
  - Matplotlib
  - Optuna
  - Pyngrok
  - Joblib
  - PyArrow

---
