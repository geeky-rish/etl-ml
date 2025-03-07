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
from pyngrok import ngrok
import warnings
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

# -------------------- ENHANCED PIPELINE -------------------- #
class ProductionETL:
    def __init__(self):
        self.preprocessor = None
        self.best_params = None

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
        df_clean.to_parquet(f"{Config.ARTIFACT_PATH}/processed_data.parquet")
        df_clean.to_csv(f"{Config.ARTIFACT_PATH}/processed_data.csv", index=False)

        return df_clean

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
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark mb-4">
        <div class="container-fluid">
            <a class="navbar-brand" href="/">AI ETL Dashboard</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/cluster">Clusters</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/distribution">Distributions</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/correlation">Correlations</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/stats">Statistics</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/download">Download</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>
    '''

@app.route('/')
def dashboard():
    return render_template_string(f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI ETL Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            .card {{ margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: transform 0.2s; }}
            .card:hover {{ transform: translateY(-5px); }}
            .plot-img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #ddd; }}
            .stats-table {{ max-height: 600px; overflow-y: auto; }}
        </style>
    </head>
    <body>
        {create_nav()}
        
        <div class="container">
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-body">
                            <h4><i class="fas fa-info-circle"></i> Dataset Overview</h4>
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
    </body>
    </html>
    ''')

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

        return render_template_string(f'''
        {create_nav()}
         <style>
            .card {{ margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: transform 0.2s; }}
            .card:hover {{ transform: translateY(-5px); }}
            .plot-img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #ddd; }}
            .stats-table {{ max-height: 600px; overflow-y: auto; }}
        </style>
        <div class="container">
            <h3 class="mb-4"><i class="fas fa-chart-pie"></i> Cluster Analysis</h3>
            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <img src="data:image/png;base64,{img1}" class="plot-img">
                        </div>
                    </div>
                </div>
                <div class="col-md-8">
                    <div class="card">
                        <div class="card-body">
                            <img src="data:image/png;base64,{img2}" class="plot-img">
                        </div>
                    </div>
                </div>
            </div>
            <div class="row mt-4">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-body">
                            <h5>3D Cluster Visualization</h5>
                            <img src="data:image/png;base64,{img3}" class="plot-img">
                        </div>
                    </div>
                </div>
            </div>
        </div>
        ''')
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

        return render_template_string(f'''
        {create_nav()}
        <style>
            .card {{ margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: transform 0.2s; }}
            .card:hover {{ transform: translateY(-5px); }}
            .plot-img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #ddd; }}
            .stats-table {{ max-height: 600px; overflow-y: auto; }}
        </style>
        <div class="container">
            <h3 class="mb-4"><i class="fas fa-chart-bar"></i> Feature Distributions</h3>
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <img src="data:image/png;base64,{img1}" class="plot-img">
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <img src="data:image/png;base64,{img2}" class="plot-img">
                        </div>
                    </div>
                </div>
            </div>
            <div class="row mt-4">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-body">
                            <img src="data:image/png;base64,{img3}" class="plot-img">
                        </div>
                    </div>
                </div>
            </div>
        </div>
        ''')
    except Exception as e:
        return error_handler(e)

@app.route('/correlation')
def correlation_analysis():
    try:
        # Correlation Matrix
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

        return render_template_string(f'''
        {create_nav()}
        <style>
            .card {{ margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: transform 0.2s; }}
            .card:hover {{ transform: translateY(-5px); }}
            .plot-img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #ddd; }}
            .stats-table {{ max-height: 600px; overflow-y: auto; }}
        </style>
        <div class="container">
            <h3 class="mb-4"><i class="fas fa-project-diagram"></i> Correlation Analysis</h3>
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-body">
                            <img src="data:image/png;base64,{img1}" class="plot-img">
                        </div>
                    </div>
                </div>
            </div>
            <div class="row mt-4">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-body">
                            <img src="data:image/png;base64,{img2}" class="plot-img">
                        </div>
                    </div>
                </div>
            </div>
        </div>
        ''')
    except Exception as e:
        return error_handler(e)

@app.route('/stats')
def statistical_analysis():
    try:
        stats = generate_stats(processed_data)
        return render_template_string(f'''
        {create_nav()}
        <style>
            .card {{ margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: transform 0.2s; }}
            .card:hover {{ transform: translateY(-5px); }}
            .plot-img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #ddd; }}
            .stats-table {{ max-height: 600px; overflow-y: auto; }}
        </style>
        <div class="container">
            <h3 class="mb-4"><i class="fas fa-calculator"></i> Statistical Analysis</h3>
            <div class="card">
                <div class="card-body stats-table">
                    {stats.to_html(classes="table table-striped table-hover")}
                </div>
            </div>
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5>Skewness Analysis</h5>
                            <ul class="list-group">
                                {"".join(f'<li class="list-group-item d-flex justify-content-between align-items-center">{col}<span class="badge bg-primary rounded-pill">{stats.loc[col,"skew"]:.2f}</span></li>' for col in stats.index)}
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5>Kurtosis Analysis</h5>
                            <ul class="list-group">
                                {"".join(f'<li class="list-group-item d-flex justify-content-between align-items-center">{col}<span class="badge bg-success rounded-pill">{stats.loc[col,"kurtosis"]:.2f}</span></li>' for col in stats.index)}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        ''')
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
    return f'''
    {create_nav()}
    <div class="container">
        <div class="alert alert-danger mt-4">
            <h4><i class="fas fa-exclamation-triangle"></i> Error Occurred</h4>
            <pre>{str(e)}</pre>
            <a href="/" class="btn btn-primary">
                <i class="fas fa-home"></i> Return to Dashboard
            </a>
        </div>
    </div>
    '''

# -------------------- MAIN EXECUTION -------------------- #
def main():
    global processed_data

    # Colab setup
    !mkdir -p {Config.ARTIFACT_PATH}
    !pip install -q flask pyngrok optuna scikit-learn pyarrow
    !ngrok authtoken 2twHHIhppxwLpGPDYSu6Gxs8zUE_4qhDG5cJsWk8H3J6wSut3
    # Run pipeline
    raw_data = pd.read_csv(Config.DATA_URL).drop(Config.DROP_FEATURES, axis=1)
    etl = ProductionETL()
    processed_data = etl.run_pipeline(raw_data)

    # Start server
    public_url = ngrok.connect(Config.PORT).public_url
    print(f"\U0001F30D Dashboard URL: {public_url}")
    app.run(host='0.0.0.0', port=Config.PORT)

if __name__ == "__main__":
    main()