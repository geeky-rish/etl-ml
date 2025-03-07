# -------------------- IMPORTS -------------------- #
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots
import joblib
import logging
import optuna
from io import BytesIO, StringIO
import base64
from datetime import datetime
from typing import Tuple, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from flask import Flask, render_template_string, send_file
import warnings
import os
from pymongo import MongoClient
import certifi
import pyarrow
import fastparquet
warnings.filterwarnings("ignore")

# -------------------- CONFIGURATION -------------------- #
class Config:
    DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch']
    CATEGORICAL_FEATURES = ['Sex', 'Embarked', 'Pclass']
    DROP_FEATURES = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    OPTUNA_TRIALS = 10
    CONTAMINATION_RANGE = (0.05, 0.2)
    N_CLUSTERS_RANGE = (2, 6)
    PORT = 5001
    ARTIFACT_PATH = "/content/artifacts"
    PLOT_STYLE = "whitegrid"
    COLOR_PALETTE = "muted"
    PLOT_SIZE = (14, 8)
    MONGO_URI = "mongodb+srv://etl_user:user123@first.t3tgs.mongodb.net/?retryWrites=true&w=majority&appName=first/"
    DB_NAME = "etl"
    COLLECTION_NAME = "etl_mk4"

# -------------------- ENHANCED PIPELINE -------------------- #

import certifi

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
uri = "mongodb+srv://etl_user:user123@first.t3tgs.mongodb.net/?retryWrites=true&w=majority&appName=first"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
class ProductionETL:
    def __init__(self):
        self.preprocessor = None
        self.best_params = None
        self.mongo_client = None
        self.db = None
        self.collection = None

    def _connect_mongo(self):
        """Connect to MongoDB Atlas"""
        try:
            self.mongo_client = MongoClient(
                Config.MONGO_URI,
                tlsCAFile=certifi.where()  # For SSL certificate verification
            )
            self.db = self.mongo_client[Config.DB_NAME]
            self.collection = self.db[Config.COLLECTION_NAME]
            print("Connected to MongoDB Atlas successfully!")
        except Exception as e:
            print(f"MongoDB connection error: {str(e)}")
            raise

    def save_csv_to_mongo(self, csv_path: str):
        """Save processed CSV file to MongoDB"""
        try:
            if not self.mongo_client:
                self._connect_mongo()

            # Debug: Check if CSV file exists
            if not os.path.exists(csv_path):
                print(f"CSV file not found: {csv_path}")
                return

            # Read the CSV file
            df = pd.read_csv(csv_path)
            print(f"Read CSV file with {len(df)} rows")

            # Convert DataFrame to dictionary (MongoDB format)
            data = df.to_dict("records")
            print(f"Converted {len(data)} rows to MongoDB documents")

            # Debug: Print the first document
            if data:
                print("First document to insert:", data[0])

            # Insert documents into MongoDB
            result = self.collection.insert_many(data)
            print(f"Inserted {len(result.inserted_ids)} documents into MongoDB")
            return result
        except Exception as e:
            print(f"Error saving CSV to MongoDB: {str(e)}")
            raise

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        self.preprocessor = self._create_preprocessor()
        best_params = self.optimize_hyperparameters(df)
        X_processed = self.preprocessor.fit_transform(df)

        iso = IsolationForest(contamination=best_params['contamination'])
        mask = iso.fit_predict(X_processed) == 1
        df_clean = df[mask].copy()

        kmeans = KMeans(n_clusters=best_params['n_clusters'], n_init=10)
        df_clean['Cluster'] = kmeans.fit_predict(self.preprocessor.transform(df_clean))

        # Save preprocessor and data locally
        joblib.dump(self.preprocessor, f"{Config.ARTIFACT_PATH}/preprocessor.joblib")
        df_clean.to_parquet(f"{Config.ARTIFACT_PATH}/processed_data.parquet", engine="fastparquet")

        # Save processed data as CSV
        csv_path = f"{Config.ARTIFACT_PATH}/processed_data.csv"
        df_clean.to_csv(csv_path, index=False)

        # Debug: Print the CSV file path
        print(f"CSV file saved at: {csv_path}")

        # Save the same CSV file to MongoDB
        self.save_csv_to_mongo(csv_path)

        return df_clean
    def _create_preprocessor(self) -> ColumnTransformer:
        numerical_pipe = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])

        categorical_pipe = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        return ColumnTransformer([
            ('num', numerical_pipe, Config.NUMERICAL_FEATURES),
            ('cat', categorical_pipe, Config.CATEGORICAL_FEATURES)
        ])

    def optimize_hyperparameters(self, X: pd.DataFrame) -> Dict[str, Any]:
        def objective(trial: optuna.Trial) -> float:
            contamination = trial.suggest_float('contamination', *Config.CONTAMINATION_RANGE)
            n_clusters = trial.suggest_int('n_clusters', *Config.N_CLUSTERS_RANGE)

            X_processed = self.preprocessor.fit_transform(X)
            iso = IsolationForest(contamination=contamination)
            mask = iso.fit_predict(X_processed) == 1
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            kmeans.fit(X_processed[mask])
            return kmeans.inertia_

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=Config.OPTUNA_TRIALS)
        self.best_params = study.best_params
        return self.best_params

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        self.preprocessor = self._create_preprocessor()
        best_params = self.optimize_hyperparameters(df)
        X_processed = self.preprocessor.fit_transform(df)

        iso = IsolationForest(contamination=best_params['contamination'])
        mask = iso.fit_predict(X_processed) == 1
        df_clean = df[mask].copy()

        kmeans = KMeans(n_clusters=best_params['n_clusters'], n_init=10)
        df_clean['Cluster'] = kmeans.fit_predict(self.preprocessor.transform(df_clean))

        joblib.dump(self.preprocessor, f"{Config.ARTIFACT_PATH}/preprocessor.joblib")
        df_clean.to_parquet(f"{Config.ARTIFACT_PATH}/processed_data.parquet", engine="fastparquet")
        df_clean.to_csv(f"{Config.ARTIFACT_PATH}/processed_data.csv", index=False)

        return df_clean
def save_csv_to_mongo():
    try:
        client = MongoClient("mongodb+srv://etl_user:user123@first.t3tgs.mongodb.net/?retryWrites=true&w=majority&appName=first")
        db = client["etl"]
        collection = db["etl_mk4"]
        file_path = f"{Config.ARTIFACT_PATH}/processed_data.csv"

        if os.path.exists(file_path):
            # Read the CSV file into a Pandas DataFrame
            df = pd.read_csv(file_path)

            # Convert DataFrame to dictionary
            data_dict = df.to_dict(orient="records")

            if data_dict:
                # Insert data into MongoDB
                collection.insert_many(data_dict)
                print("✅ Processed data successfully saved to MongoDB!")
            else:
                print("⚠️ CSV file is empty. No data to save.")
        else:
            print("❌ CSV file not found!")

    except Exception as e:
        print(f"Error saving to MongoDB: {e}")

# -------------------- PROFESSIONAL DASHBOARD -------------------- #
app = Flask(__name__)
sns.set_style(Config.PLOT_STYLE)
sns.set_palette(Config.COLOR_PALETTE)
processed_data = None

def create_plot():
    plt.figure(figsize=Config.PLOT_SIZE)
    return plt

def render_plot(plt):
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_stats(df):
    try:
        # Convert numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Calculate statistics
        stats = numeric_df.describe().T
        stats['skew'] = numeric_df.skew()
        stats['kurtosis'] = numeric_df.kurtosis()
        stats['missing'] = numeric_df.isnull().sum()

        return stats
    except Exception as e:
        print(f"Error generating stats: {e}")
        return pd.DataFrame()

def create_nav():
    return '''
    <nav class="navbar">
        <div class="container">
            <a class="navbar-brand" href="/">AI ETL Dashboard</a>
            <button class="navbar-toggler" type="button" id="navToggle">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="navbar-menu" id="navbarMenu">
                <ul class="navbar-nav">
                    <li class="nav-item"><a class="nav-link" href="/cluster">Clusters</a></li>
                    <li class="nav-item"><a class="nav-link" href="/distribution">Distributions</a></li>
                    <li class="nav-item"><a class="nav-link" href="/correlation">Correlations</a></li>
                    <li class="nav-item"><a class="nav-link" href="/stats">Statistics</a></li>
                    <li class="nav-item"><a class="nav-link" href="/download">Download</a></li>
                </ul>
            </div>
        </div>
    </nav>
    <script>
        document.getElementById('navToggle').addEventListener('click', function() {
            document.getElementById('navbarMenu').classList.toggle('active');
        });
    </script>
    '''

def create_base_template():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI ETL Dashboard</title>
        <style>
            :root {
                --primary: #4361ee;
                --secondary: #3f37c9;
                --success: #4cc9f0;
                --info: #4895ef;
                --warning: #f72585;
                --danger: #e63946;
                --light: #f8f9fa;
                --dark: #212529;
                --gray: #6c757d;
                --gray-dark: #343a40;
                --gray-light: #e9ecef;
                --body-bg: #f5f7fa;
                --card-bg: #ffffff;
                --text-main: #333333;
                --text-light: #6c757d;
                --border-radius: 0.5rem;
                --box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                --transition: all 0.3s ease;
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: var(--text-main);
                background-color: var(--body-bg);
                padding-bottom: 2rem;
            }

            .container {
                width: 100%;
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 1rem;
            }

            /* Navbar Styles */
            .navbar {
                background-color: var(--dark);
                color: white;
                padding: 1rem 0;
                margin-bottom: 2rem;
                box-shadow: var(--box-shadow);
            }

            .navbar .container {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .navbar-brand {
                font-size: 1.5rem;
                font-weight: 700;
                color: white;
                text-decoration: none;
            }

            .navbar-toggler {
                display: none;
                background: none;
                border: none;
                color: white;
                font-size: 1.5rem;
                cursor: pointer;
            }

            .navbar-nav {
                display: flex;
                list-style: none;
                gap: 1.5rem;
            }

            .nav-link {
                color: rgba(255, 255, 255, 0.8);
                text-decoration: none;
                font-weight: 500;
                transition: var(--transition);
                padding: 0.5rem 0;
                position: relative;
            }

            .nav-link:hover {
                color: white;
            }

            .nav-link::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                width: 0;
                height: 2px;
                background-color: var(--primary);
                transition: var(--transition);
            }

            .nav-link:hover::after {
                width: 100%;
            }

            /* Card Styles */
            .card {
                background-color: var(--card-bg);
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                margin-bottom: 1.5rem;
                overflow: hidden;
                transition: var(--transition);
            }

            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12);
            }

            .card-body {
                padding: 1.5rem;
            }

            .card-title {
                font-size: 1.25rem;
                margin-bottom: 1rem;
                color: var(--dark);
                font-weight: 600;
            }

            /* Grid System */
            .row {
                display: flex;
                flex-wrap: wrap;
                margin: 0 -0.75rem;
            }

            .col-md-12, .col-md-6, .col-md-4, .col-md-8 {
                padding: 0 0.75rem;
                width: 100%;
            }

            /* Alert Styles */
            .alert {
                padding: 1rem;
                border-radius: var(--border-radius);
                margin-bottom: 1rem;
            }

            .alert-primary {
                background-color: rgba(67, 97, 238, 0.1);
                border-left: 4px solid var(--primary);
                color: var(--primary);
            }

            .alert-success {
                background-color: rgba(76, 201, 240, 0.1);
                border-left: 4px solid var(--success);
                color: var(--success);
            }

            .alert-info {
                background-color: rgba(72, 149, 239, 0.1);
                border-left: 4px solid var(--info);
                color: var(--info);
            }

            .alert-danger {
                background-color: rgba(230, 57, 70, 0.1);
                border-left: 4px solid var(--danger);
                color: var(--danger);
            }

            /* Button Styles */
            .btn {
                display: inline-block;
                font-weight: 500;
                text-align: center;
                white-space: nowrap;
                vertical-align: middle;
                user-select: none;
                border: 1px solid transparent;
                padding: 0.5rem 1rem;
                font-size: 1rem;
                line-height: 1.5;
                border-radius: var(--border-radius);
                transition: var(--transition);
                cursor: pointer;
                text-decoration: none;
            }

            .btn-primary {
                background-color: var(--primary);
                color: white;
            }

            .btn-primary:hover {
                background-color: var(--secondary);
            }

            /* Table Styles */
            .table {
                width: 100%;
                margin-bottom: 1rem;
                color: var(--text-main);
                border-collapse: collapse;
            }

            .table th,
            .table td {
                padding: 0.75rem;
                vertical-align: top;
                border-top: 1px solid var(--gray-light);
            }

            .table thead th {
                vertical-align: bottom;
                border-bottom: 2px solid var(--gray-light);
                background-color: var(--gray-light);
                color: var(--dark);
                font-weight: 600;
            }

            .table-striped tbody tr:nth-of-type(odd) {
                background-color: rgba(0, 0, 0, 0.02);
            }

            .table-hover tbody tr:hover {
                background-color: rgba(0, 0, 0, 0.04);
            }

            /* List Group */
            .list-group {
                display: flex;
                flex-direction: column;
                padding-left: 0;
                margin-bottom: 0;
                border-radius: var(--border-radius);
                overflow: hidden;
            }

            .list-group-item {
                position: relative;
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.75rem 1.25rem;
                background-color: var(--card-bg);
                border: 1px solid rgba(0, 0, 0, 0.125);
                border-width: 0 0 1px;
            }

            .list-group-item:last-child {
                border-bottom-width: 0;
            }

            .badge {
                display: inline-block;
                padding: 0.35em 0.65em;
                font-size: 0.75em;
                font-weight: 700;
                line-height: 1;
                text-align: center;
                white-space: nowrap;
                vertical-align: baseline;
                border-radius: 50rem;
                color: white;
            }

            .badge-primary {
                background-color: var(--primary);
            }

            .badge-success {
                background-color: var(--success);
            }

            /* Image Styles */
            .plot-img {
                width: 100%;
                height: auto;
                border-radius: var(--border-radius);
                border: 1px solid var(--gray-light);
                transition: var(--transition);
            }

            .plot-img:hover {
                transform: scale(1.01);
                box-shadow: var(--box-shadow);
            }

            /* Utilities */
            .mb-4 {
                margin-bottom: 1.5rem;
            }

            .mt-4 {
                margin-top: 1.5rem;
            }

            .stats-table {
                max-height: 600px;
                overflow-y: auto;
            }

            h3, h4, h5 {
                color: var(--dark);
                margin-bottom: 1rem;
                font-weight: 600;
            }

            h3 {
                font-size: 1.75rem;
            }

            h4 {
                font-size: 1.5rem;
            }

            h5 {
                font-size: 1.25rem;
            }

            /* Responsive Styles */
            @media (min-width: 768px) {
                .col-md-12 {
                    width: 100%;
                }
                .col-md-8 {
                    width: 66.666667%;
                }
                .col-md-6 {
                    width: 50%;
                }
                .col-md-4 {
                    width: 33.333333%;
                }
            }

            @media (max-width: 767px) {
                .navbar-toggler {
                    display: block;
                }
                .navbar-menu {
                    position: fixed;
                    top: 0;
                    right: -250px;
                    width: 250px;
                    height: 100vh;
                    background-color: var(--dark);
                    padding: 2rem 1rem;
                    transition: var(--transition);
                    z-index: 1000;
                    box-shadow: -5px 0 15px rgba(0, 0, 0, 0.1);
                }
                .navbar-menu.active {
                    right: 0;
                }
                .navbar-nav {
                    flex-direction: column;
                    gap: 1rem;
                }
                .card {
                    margin-bottom: 1rem;
                }
            }
        </style>
    </head>
    <body>
        {content}
        <script>
            // Add any JavaScript here
            document.addEventListener('DOMContentLoaded', function() {
                // Add smooth scrolling
                document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                    anchor.addEventListener('click', function (e) {
                        e.preventDefault();
                        document.querySelector(this.getAttribute('href')).scrollIntoView({
                            behavior: 'smooth'
                        });
                    });
                });
            });
        </script>
    </body>
    </html>
    '''

@app.route('/')
def dashboard():
    content = f'''
    {create_nav()}
    <div class="container">
        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-body">
                        <h4 class="card-title">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <circle cx="12" cy="12" r="10"></circle>
                                <line x1="12" y1="16" x2="12" y2="12"></line>
                                <line x1="12" y1="8" x2="12.01" y2="8"></line>
                            </svg>
                            Dataset Overview
                        </h4>
                        <div class="row">
                            <div class="col-md-4">
                                <div class="alert alert-primary">
                                    <h5>Total Samples</h5>
                                    <h3>{len(processed_data):,}</h3>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="alert alert-success">
                                    <h5>Features</h5>
                                    <h3>{len(processed_data.columns)}</h3>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="alert alert-info">
                                    <h5>Clusters</h5>
                                    <h3>{processed_data['Cluster'].nunique()}</h3>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    '''
    return render_template_string(create_base_template().replace('{content}', content))

@app.route('/cluster')
def cluster_analysis():
    try:
        # Cluster Distribution
        plt1 = create_plot()
        sns.countplot(x='Cluster', data=processed_data)
        plt.title('Cluster Distribution')
        img1 = render_plot(plt1)

        # Cluster Characteristics
        plt2 = create_plot()
        sns.boxplot(x='Cluster', y='Age', data=processed_data)
        plt.title('Age Distribution by Cluster')
        img2 = render_plot(plt2)

        # 3D Visualization
        plt3 = create_plot()
        ax = plt3.figure().add_subplot(111, projection='3d')  # Fixed 3D plot
        ax.scatter(
            processed_data['Age'],
            processed_data['Fare'],
            processed_data['SibSp'],
            c=processed_data['Cluster'],
            cmap='viridis'
        )
        ax.set_xlabel('Age')
        ax.set_ylabel('Fare')
        ax.set_zlabel('Siblings/Spouses')
        img3 = render_plot(plt3)

        content = f'''
        {create_nav()}
        <div class="container">
            <h3 class="mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path>
                    <path d="M22 12A10 10 0 0 0 12 2v10z"></path>
                </svg>
                Cluster Analysis
            </h3>
            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Cluster Distribution</h5>
                            <img src="data:image/png;base64,{img1}" class="plot-img">
                        </div>
                    </div>
                </div>
                <div class="col-md-8">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Age Distribution by Cluster</h5>
                            <img src="data:image/png;base64,{img2}" class="plot-img">
                        </div>
                    </div>
                </div>
            </div>
            <div class="row mt-4">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">3D Cluster Visualization</h5>
                            <img src="data:image/png;base64,{img3}" class="plot-img">
                        </div>
                    </div>
                </div>
            </div>
        </div>
        '''
        return render_template_string(create_base_template().replace('{content}', content))
    except Exception as e:
        return error_handler(e)

@app.route('/distribution')
def distribution_analysis():
    try:
        # Age Distribution
        plt1 = create_plot()
        sns.histplot(processed_data['Age'], kde=True, bins=30)
        plt.title('Age Distribution')
        img1 = render_plot(plt1)

        # Fare Distribution
        plt2 = create_plot()
        sns.boxplot(x=processed_data['Fare'])
        plt.title('Fare Distribution')
        img2 = render_plot(plt2)

        # Combined Distribution
        plt3 = create_plot()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        sns.histplot(processed_data['SibSp'], kde=True, ax=axes[0,0])
        sns.histplot(processed_data['Parch'], kde=True, ax=axes[0,1])
        sns.countplot(x='Sex', data=processed_data, ax=axes[1,0])
        sns.countplot(x='Pclass', data=processed_data, ax=axes[1,1])
        plt.tight_layout()
        img3 = render_plot(plt3)

        content = f'''
        {create_nav()}
        <div class="container">
            <h3 class="mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M18 20V10"></path>
                    <path d="M12 20V4"></path>
                    <path d="M6 20v-6"></path>
                </svg>
                Feature Distributions
            </h3>
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Age Distribution</h5>
                            <img src="data:image/png;base64,{img1}" class="plot-img">
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Fare Distribution</h5>
                            <img src="data:image/png;base64,{img2}" class="plot-img">
                        </div>
                    </div>
                </div>
            </div>
            <div class="row mt-4">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Combined Feature Distributions</h5>
                            <img src="data:image/png;base64,{img3}" class="plot-img">
                        </div>
                    </div>
                </div>
            </div>
        </div>
        '''
        return render_template_string(create_base_template().replace('{content}', content))
    except Exception as e:
        return error_handler(e)

@app.route('/correlation')
def correlation_analysis():
    try:
        plt1 = create_plot()
        corr = processed_data.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title('Feature Correlation Matrix')
        img1 = render_plot(plt1)

        # Cluster Correlation
        plt2 = create_plot()
        cluster_corr = processed_data.groupby('Cluster').mean().corr()
        sns.heatmap(cluster_corr, annot=True, fmt=".2f", cmap="viridis")
        plt.title('Cluster-wise Correlation')
        img2 = render_plot(plt2)

        content = f'''
        {create_nav()}
        <div class="container">
            <h3 class="mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="3" y1="12" x2="21" y2="12"></line>
                    <line x1="3" y1="6" x2="21" y2="6"></line>
                    <line x1="3" y1="18" x2="21" y2="18"></line>
                </svg>
                Correlation Analysis
            </h3>
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Feature Correlation Matrix</h5>
                            <img src="data:image/png;base64,{img1}" class="plot-img">
                        </div>
                    </div>
                </div>
            </div>
            <div class="row mt-4">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Cluster-wise Correlation</h5>
                            <img src="data:image/png;base64,{img2}" class="plot-img">
                        </div>
                    </div>
                </div>
            </div>
        </div>
        '''
        return render_template_string(create_base_template().replace('{content}', content))
    except Exception as e:
        return error_handler(e)

@app.route('/stats')
def statistical_analysis():
    try:
        stats = generate_stats(processed_data)
        content = f'''
        {create_nav()}
        <div class="container">
            <h3 class="mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
                    <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
                </svg>
                Statistical Analysis
            </h3>
            <div class="card">
                <div class="card-body stats-table">
                    <h5 class="card-title">Descriptive Statistics</h5>
                    {stats.to_html(classes="table table-striped table-hover")}
                </div>
            </div>
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Skewness Analysis</h5>
                            <ul class="list-group">
                                {"".join(f'<li class="list-group-item d-flex justify-content-between align-items-center">{col}<span class="badge badge-primary">{stats.loc[col,"skew"]:.2f}</span></li>' for col in stats.index)}
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Kurtosis Analysis</h5>
                            <ul class="list-group">
                                {"".join(f'<li class="list-group-item d-flex justify-content-between align-items-center">{col}<span class="badge badge-success">{stats.loc[col,"kurtosis"]:.2f}</span></li>' for col in stats.index)}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        '''
        return render_template_string(create_base_template().replace('{content}', content))
    except Exception as e:
        return error_handler(e)

@app.route('/download')
def download_data():
    try:
        return send_file(
            f"{Config.ARTIFACT_PATH}/processed_data.csv",
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"processed_data_{datetime.now().strftime('%Y%m%d')}.csv"
        )
    except Exception as e:
        return error_handler(e)

def error_handler(e):
    content = f'''
    {create_nav()}
    <div class="container">
        <div class="alert alert-danger mt-4">
            <h4>
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                    <line x1="12" y1="9" x2="12" y2="13"></line>
                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
                Error Occurred
            </h4>
            <pre style="background-color: rgba(230, 57, 70, 0.05); padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">{str(e)}</pre>
            <a href="/" class="btn btn-primary mt-4">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.5rem;">
                    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                    <polyline points="9 22 9 12 15 12 15 22"></polyline>
                </svg>
                Return to Dashboard
            </a>
        </div>
    </div>
    '''
    return render_template_string(create_base_template().replace('{content}', content))

# -------------------- MAIN EXECUTION -------------------- #
def main():
    global processed_data
    os.makedirs(Config.ARTIFACT_PATH, exist_ok=True)
    save_csv_to_mongo()
    # Run pipeline
    raw_data = pd.read_csv(Config.DATA_URL).drop(Config.DROP_FEATURES, axis=1)
    etl = ProductionETL()
    processed_data = etl.run_pipeline(raw_data)
    app.run(host='0.0.0.0', port=Config.PORT)

if __name__ == "__main__":
    main()
