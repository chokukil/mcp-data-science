"""
Advanced Data Science MCP Server - Enhanced Version
고도화된 데이터 분석, 시각화, 머신러닝, 딥러닝 기능을 제공하는 MCP 서버
AutoKeras, 고급 알고리즘, 인터랙티브 시각화, 모델 해석 기능 포함
실행한 코드도 산출물로 제공
"""

import os, sys, json, logging, argparse, uuid, traceback, warnings
import asyncio, uvicorn, shutil, ast, textwrap, glob, re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger = logging.getLogger(__name__)
    logger.info("Environment variables loaded from .env file")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("python-dotenv not available, skipping .env file loading")

# ChatGPT-style LLM 호출용
try:
    import openai
    # Set OpenAI API key from environment variable
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        openai.api_key = openai_api_key
        logger.info("OpenAI API key loaded from environment")
    else:
        logger.warning("OPENAI_API_KEY not found in environment variables")
except ImportError:
    openai = None        # MCP venv 에 openai 패키지가 없을 경우 대비
    logger.warning("openai package not available")

# Core data science packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Interactive visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not available - interactive visualizations disabled")

# Machine Learning packages
try:
    import sklearn
    from sklearn import datasets, metrics, model_selection, preprocessing
    from sklearn import linear_model, ensemble, cluster, neural_network
    from sklearn import svm, tree, naive_bayes, neighbors, decomposition
    from sklearn import feature_selection, pipeline, mixture
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
    from sklearn.decomposition import KernelPCA, FastICA, FactorAnalysis, LatentDirichletAllocation
    from sklearn.cluster import OPTICS, MeanShift, AffinityPropagation, SpectralClustering
    from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# AutoKeras for automated deep learning
try:
    import autokeras as ak
    AUTOKERAS_AVAILABLE = True
except ImportError:
    AUTOKERAS_AVAILABLE = False

# Keras Tuner for hyperparameter optimization
try:
    import keras_tuner as kt
    KERAS_TUNER_AVAILABLE = True
except ImportError:
    KERAS_TUNER_AVAILABLE = False

# Advanced boosting models
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Time series analysis
try:
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Model interpretation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Advanced dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Advanced clustering
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# Imbalanced data handling
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
    from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
    from imblearn.combine import SMOTEENN, SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# MCP Server
from mcp.server.fastmcp import FastMCP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_sandbox_directory() -> Tuple[str, int]:
    """
    Determines the sandbox directory for data science operations.
    
    Returns:
        Tuple[str, int]: Path to sandbox directory and server port
    """
    parser = argparse.ArgumentParser(description='Advanced Data Science MCP Server')
    parser.add_argument('--sandbox-dir', type=str, default='../../sandbox', 
                       help='Sandbox directory for data science operations')
    parser.add_argument('--port', type=int, default=8007, 
                       help='Port to run the server on (default: 8007)')
    args, _ = parser.parse_known_args()
    
    # Get absolute path for sandbox directory
    if args.sandbox_dir:
        sandbox_dir = os.path.abspath(args.sandbox_dir)
        logger.info(f"Using sandbox directory: {sandbox_dir}")
        return sandbox_dir, args.port
    
    # Default sandbox directory
    sandbox_dir = os.path.abspath('../../sandbox')
    logger.info(f"Using default sandbox directory: {sandbox_dir}")
    return sandbox_dir, args.port

# Get sandbox directory and server port
SANDBOX_DIR, SERVER_PORT = get_sandbox_directory()

# Ensure sandbox directory exists
if not os.path.exists(SANDBOX_DIR):
    try:
        os.makedirs(SANDBOX_DIR, exist_ok=True)
        logger.info(f"Created sandbox directory: {SANDBOX_DIR}")
    except Exception as e:
        logger.error(f"Failed to create sandbox directory {SANDBOX_DIR}: {e}")
        sys.exit(1)

# Create subdirectories for organized storage
DATASETS_DIR = os.path.join(SANDBOX_DIR, 'datasets')
PLOTS_DIR = os.path.join(SANDBOX_DIR, 'plots')
MODELS_DIR = os.path.join(SANDBOX_DIR, 'models')
REPORTS_DIR = os.path.join(SANDBOX_DIR, 'reports')
LOGS_DIR = os.path.join(SANDBOX_DIR, 'logs')
CODE_DIR = os.path.join(SANDBOX_DIR, 'generated_code')

for dir_path in [DATASETS_DIR, PLOTS_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR, CODE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

logger.info(f"Advanced data science operations will be performed in: {SANDBOX_DIR}")
logger.info(f"Server will run on port: {SERVER_PORT}")

# Global state management for tracking operations
class AdvancedOperationTracker:
    """Enhanced tracker for data science operations with interpretability"""
    
    def __init__(self):
        self.operations = {}
        self.datasets = {}
        self.models = {}
        self.plots = {}
        self.experiments = {}
    
    def log_operation(self, operation_id: str, operation_type: str, 
                     input_data: Dict, output_data: Dict, 
                     files_created: List[str] = None,
                     metrics: Dict = None):
        """Log an operation for tracking with optional metrics"""
        self.operations[operation_id] = {
            'id': operation_id,
            'type': operation_type,
            'timestamp': datetime.now().isoformat(),
            'input': input_data,
            'output': output_data,
            'files_created': files_created or [],
            'metrics': metrics or {},
            'status': 'completed'
        }
        
        try:
            # Ensure logs directory exists
            ensure_directories_exist(LOGS_DIR)
            
            # Save operation log
            log_file = os.path.join(LOGS_DIR, f'operation_{operation_id}.json')
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(self.operations[operation_id], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save operation log for {operation_id}: {e}")
    
    def get_operation(self, operation_id: str) -> Optional[Dict]:
        """Get operation details by ID"""
        return self.operations.get(operation_id)
    
    def register_dataset(self, dataset_id: str, df: pd.DataFrame, source: str,
                        sampling_info: Dict = None, metadata: Dict = None):
        """Register a dataset with enhanced metadata"""
        self.datasets[dataset_id] = {
            'id': dataset_id,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'sampling_info': sampling_info or {},
            'metadata': metadata or {},
            'statistics': {
                'numeric_columns': len(df.select_dtypes(include=['number']).columns),
                'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
                'missing_ratio': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            }
        }
    
    def register_model(self, model_id: str, model_info: Dict):
        """Register a model"""
        self.models[model_id] = {
            'id': model_id,
            'timestamp': datetime.now().isoformat(),
            **model_info
        }
    
    def register_plot(self, plot_id: str, plot_path: str, plot_info: Dict):
        """Register a plot"""
        self.plots[plot_id] = {
            'id': plot_id,
            'path': plot_path,
            'timestamp': datetime.now().isoformat(),
            **plot_info
        }
    
    def register_experiment(self, experiment_id: str, experiment_info: Dict):
        """Register an experiment with multiple models"""
        self.experiments[experiment_id] = {
            'id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            **experiment_info
        }

# Global tracker instance
tracker = AdvancedOperationTracker()

def save_plot_as_image(operation_id: str, plot_title: str = "plot") -> str:
    """
    Save current matplotlib plot as image and return file path
    
    Args:
        operation_id: Operation ID for file naming
        plot_title: Title for the plot file
        
    Returns:
        str: Path to saved plot file
    """
    try:
        # Ensure plots directory exists
        ensure_directories_exist(PLOTS_DIR)
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_id = f"{operation_id}_{plot_title}_{timestamp}"
        plot_filename = f"{plot_id}.png"
        plot_path = os.path.join(PLOTS_DIR, plot_filename)
        
        # Save the current figure
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()  # Close to free memory
        
        # Register plot in tracker
        tracker.register_plot(plot_id, plot_path, {
            'title': plot_title,
            'operation_id': operation_id,
            'created_at': datetime.now().isoformat()
        })
        
        logger.info(f"Plot saved: {plot_path}")
        return plot_path
        
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
        plt.close()  # Ensure plot is closed even on error
        return ""

def ensure_directories_exist(*dirs):
    """
    Ensure that the specified directories exist, creating them if necessary.
    
    Args:
        *dirs: Variable number of directory paths to check/create
        
    Returns:
        bool: True if all directories exist or were created successfully
        
    Raises:
        Exception: If directory creation fails
    """
    for dir_path in dirs:
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            elif not os.path.isdir(dir_path):
                # Path exists but is not a directory
                raise OSError(f"Path exists but is not a directory: {dir_path}")
        except PermissionError:
            raise Exception(f"Permission denied: Cannot create directory {dir_path}")
        except OSError as e:
            raise Exception(f"Failed to create directory {dir_path}: {str(e)}")
    
    return True

def ensure_sandbox_structure():
    """
    Ensure all sandbox subdirectories exist.
    This function is called at the start of each tool to guarantee directory structure.
    
    Returns:
        bool: True if all directories exist or were created successfully
    """
    try:
        ensure_directories_exist(
            SANDBOX_DIR,
            DATASETS_DIR, 
            PLOTS_DIR, 
            MODELS_DIR, 
            REPORTS_DIR, 
            LOGS_DIR,
            CODE_DIR
        )
        return True
    except Exception as e:
        logger.error(f"Failed to ensure sandbox structure: {e}")
        return False

def safe_json_serialize(obj):
    """
    Safely serialize objects to JSON-compatible format
    """
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return json.loads(obj.to_json())
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif hasattr(obj, 'dtype'):
        # Handle pandas extension dtypes
        return str(obj)
    else:
        return obj

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except:
        return 0.0

def auto_detect_problem_type(df: pd.DataFrame, target_column: str = None) -> str:
    """
    Automatically detect the problem type based on data characteristics
    
    Args:
        df: DataFrame to analyze
        target_column: Target column name if provided
        
    Returns:
        str: Detected problem type
    """
    if target_column is None:
        # Check for time series data
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            return 'timeseries'
        
        # Check for date/time patterns in column names
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp', 'year', 'month']):
                return 'timeseries'
        
        # Check for image data
        if any('image' in col.lower() or 'img' in col.lower() or 'photo' in col.lower() 
               for col in df.columns):
            return 'image_analysis'
        
        # Check for text data
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            sample_values = df[col].dropna().astype(str).head(10)
            avg_length = sample_values.str.len().mean()
            if avg_length > 50:  # Average length > 50 chars suggests text
                return 'text_analysis'
        
        return 'clustering'
    
    if target_column not in df.columns:
        return 'clustering'
    
    target = df[target_column]
    
    # Check for time series forecasting
    if df.index.dtype.kind in 'Mm' or any(col.lower() in ['date', 'time', 'timestamp'] for col in df.columns):
        return 'timeseries_forecasting'
    
    # Check for image classification/regression
    if any('image' in col.lower() or 'img' in col.lower() for col in df.columns):
        if target.dtype.kind in 'biufc':
            unique_ratio = len(target.unique()) / len(target)
            return 'image_regression' if unique_ratio > 0.05 else 'image_classification'
        else:
            return 'image_classification'
    
    # Check for text classification
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        if col != target_column:
            sample_values = df[col].dropna().astype(str).head(10)
            avg_length = sample_values.str.len().mean()
            if avg_length > 50:
                return 'text_classification'
    
    # Standard classification/regression
    if target.dtype.kind in 'biufc':
        unique_ratio = len(target.unique()) / len(target)
        if unique_ratio > 0.05 or len(target.unique()) > 20:
            return 'regression'
        else:
            return 'classification'
    
    return 'classification'

def determine_sampling_strategy(df: pd.DataFrame, file_size_mb: float, target_column: str = None) -> Dict:
    """
    Determine the best sampling strategy based on data characteristics
    
    Args:
        df: DataFrame to analyze
        file_size_mb: File size in megabytes
        target_column: Target column for stratified sampling
        
    Returns:
        Dict: Sampling strategy information
    """
    n_rows = len(df)
    n_cols = len(df.columns)
    
    # 샘플링 필요성 판단
    if file_size_mb < 100:
        return {
            "sampling_needed": False,
            "reason": f"File size ({file_size_mb:.1f}MB) is under 100MB threshold",
            "method": None,
            "sample_size": n_rows,
            "sample_ratio": 1.0
        }
    
    # 목표 샘플 크기 결정 (메모리와 처리 시간 고려)
    if file_size_mb < 500:  # 100-500MB
        target_rows = min(50000, n_rows)
    elif file_size_mb < 1000:  # 500MB-1GB
        target_rows = min(30000, n_rows)
    elif file_size_mb < 2000:  # 1-2GB
        target_rows = min(20000, n_rows)
    else:  # 2GB+
        target_rows = min(15000, n_rows)
    
    sample_ratio = target_rows / n_rows
    
    # 샘플링 방법 결정
    if target_column and target_column in df.columns:
        # 타겟 컬럼이 있으면 stratified sampling
        method = "stratified"
        reason = f"Using stratified sampling to preserve target variable '{target_column}' distribution"
    elif n_rows > 1000000:  # 100만 행 이상
        # 매우 큰 데이터는 systematic sampling
        method = "systematic"
        reason = "Using systematic sampling for very large dataset to ensure representative coverage"
    else:
        # 기본적으로 random sampling
        method = "random"
        reason = "Using random sampling to maintain data randomness"
    
    return {
        "sampling_needed": True,
        "reason": f"File size ({file_size_mb:.1f}MB) exceeds 100MB threshold. Sampling to improve processing speed and memory usage.",
        "method": method,
        "sample_size": target_rows,
        "sample_ratio": sample_ratio,
        "original_size": n_rows,
        "strategy_reason": reason,
        "estimated_memory_reduction": f"{(1-sample_ratio)*100:.1f}%"
    }

def apply_sampling(df: pd.DataFrame, sampling_info: Dict, target_column: str = None, random_state: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply sampling to DataFrame based on sampling strategy
    
    Args:
        df: Original DataFrame
        sampling_info: Sampling strategy information
        target_column: Target column for stratified sampling
        random_state: Random state for reproducibility
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Sampled DataFrame and sampling details
    """
    if not sampling_info["sampling_needed"]:
        return df, {"applied": False, "message": "No sampling applied"}
    
    method = sampling_info["method"]
    sample_size = sampling_info["sample_size"]
    
    try:
        if method == "stratified" and target_column and target_column in df.columns:
            # Stratified sampling
            from sklearn.model_selection import train_test_split
            
            # 각 클래스별로 최소 샘플 수 확보
            target_counts = df[target_column].value_counts()
            min_samples_per_class = max(1, sample_size // len(target_counts))
            
            # 클래스별 샘플링 비율 조정
            if target_counts.min() < min_samples_per_class:
                # 일부 클래스가 너무 적으면 random sampling으로 대체
                sampled_df = df.sample(n=sample_size, random_state=random_state)
                actual_method = "random (fallback from stratified)"
                details = f"Switched to random sampling due to insufficient samples in some classes"
            else:
                # Stratified sampling 실행
                _, sampled_df = train_test_split(
                    df, test_size=sample_size, random_state=random_state,
                    stratify=df[target_column]
                )
                actual_method = "stratified"
                details = f"Maintained target variable distribution across {len(target_counts)} classes"
                
        elif method == "systematic":
            # Systematic sampling
            step = len(df) // sample_size
            indices = list(range(0, len(df), step))[:sample_size]
            sampled_df = df.iloc[indices].copy()
            actual_method = "systematic"
            details = f"Selected every {step}th row for systematic coverage"
            
        else:  # random sampling
            sampled_df = df.sample(n=sample_size, random_state=random_state)
            actual_method = "random"
            details = f"Randomly selected {sample_size} rows"
        
        # 샘플링 품질 검증
        quality_check = validate_sampling_quality(df, sampled_df, target_column)
        
        sampling_details = {
            "applied": True,
            "method": actual_method,
            "original_rows": len(df),
            "sampled_rows": len(sampled_df),
            "sample_ratio": len(sampled_df) / len(df),
            "details": details,
            "quality_check": quality_check,
            "random_state": random_state
        }
        
        return sampled_df, sampling_details
        
    except Exception as e:
        logger.warning(f"Sampling failed: {e}. Using first {sample_size} rows instead.")
        # 샘플링 실패 시 상위 N개 행 사용
        sampled_df = df.head(sample_size)
        sampling_details = {
            "applied": True,
            "method": "head (fallback)",
            "original_rows": len(df),
            "sampled_rows": len(sampled_df),
            "sample_ratio": len(sampled_df) / len(df),
            "details": f"Fallback to first {sample_size} rows due to sampling error: {str(e)}",
            "quality_check": {"warning": "Quality check skipped due to fallback method"}
        }
        return sampled_df, sampling_details

def validate_sampling_quality(original_df: pd.DataFrame, sampled_df: pd.DataFrame, target_column: str = None) -> Dict:
    """
    Validate the quality of sampling by comparing distributions
    
    Args:
        original_df: Original DataFrame
        sampled_df: Sampled DataFrame  
        target_column: Target column to validate distribution
        
    Returns:
        Dict: Quality validation results
    """
    quality_check = {}
    
    try:
        # 기본 통계 비교
        numeric_cols = original_df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            orig_stats = original_df[numeric_cols].describe()
            sample_stats = sampled_df[numeric_cols].describe()
            
            # 평균값 차이 계산
            mean_differences = []
            for col in numeric_cols:
                if col in sample_stats.columns:
                    orig_mean = orig_stats.loc['mean', col]
                    sample_mean = sample_stats.loc['mean', col]
                    if orig_mean != 0:
                        diff_pct = abs((sample_mean - orig_mean) / orig_mean) * 100
                        mean_differences.append(diff_pct)
            
            if mean_differences:
                avg_mean_diff = np.mean(mean_differences)
                quality_check["numeric_similarity"] = {
                    "average_mean_difference_pct": round(avg_mean_diff, 2),
                    "quality_rating": "excellent" if avg_mean_diff < 5 else "good" if avg_mean_diff < 15 else "fair"
                }
        
        # 타겟 변수 분포 비교 (분류 문제인 경우)
        if target_column and target_column in original_df.columns:
            orig_dist = original_df[target_column].value_counts(normalize=True).sort_index()
            sample_dist = sampled_df[target_column].value_counts(normalize=True).sort_index()
            
            # 공통 클래스에 대한 분포 차이 계산
            common_classes = set(orig_dist.index) & set(sample_dist.index)
            if common_classes:
                dist_differences = []
                for cls in common_classes:
                    diff = abs(orig_dist[cls] - sample_dist[cls])
                    dist_differences.append(diff)
                
                avg_dist_diff = np.mean(dist_differences) * 100
                quality_check["target_distribution"] = {
                    "average_distribution_difference_pct": round(avg_dist_diff, 2),
                    "classes_preserved": len(common_classes),
                    "total_original_classes": len(orig_dist),
                    "quality_rating": "excellent" if avg_dist_diff < 2 else "good" if avg_dist_diff < 5 else "fair"
                }
        
        # 전체 품질 평가
        ratings = []
        if "numeric_similarity" in quality_check:
            ratings.append(quality_check["numeric_similarity"]["quality_rating"])
        if "target_distribution" in quality_check:
            ratings.append(quality_check["target_distribution"]["quality_rating"])
        
        if ratings:
            if all(r == "excellent" for r in ratings):
                overall_quality = "excellent"
            elif any(r == "excellent" for r in ratings) or all(r == "good" for r in ratings):
                overall_quality = "good"
            else:
                overall_quality = "fair"
            
            quality_check["overall_quality"] = overall_quality
        
    except Exception as e:
        quality_check["error"] = f"Quality check failed: {str(e)}"
    
    return quality_check

def get_optimal_cluster_number(X: np.ndarray, max_clusters: int = 10) -> Tuple[int, Dict]:
    """
    Determine optimal number of clusters using silhouette analysis
    
    Args:
        X: Feature matrix
        max_clusters: Maximum number of clusters to test
        
    Returns:
        Tuple[int, Dict]: Optimal cluster number and analysis details
    """
    from sklearn.metrics import silhouette_score
    
    if len(X) < 4:
        return 2, {"method": "minimum", "scores": {}}
    
    max_clusters = min(max_clusters, len(X) // 2)
    silhouette_scores = {}
    inertia_scores = {}
    
    for n_clusters in range(2, max_clusters + 1):
        try:
            kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores[n_clusters] = silhouette_avg
            inertia_scores[n_clusters] = kmeans.inertia_
            
        except Exception as e:
            logger.warning(f"Failed to compute silhouette for {n_clusters} clusters: {e}")
            continue
    
    if not silhouette_scores:
        return 2, {"method": "fallback", "scores": {}}
    
    # Select optimal number of clusters
    optimal_clusters = max(silhouette_scores.keys(), key=lambda k: silhouette_scores[k])
    
    analysis = {
        "method": "silhouette",
        "silhouette_scores": silhouette_scores,
        "inertia_scores": inertia_scores,
        "optimal_clusters": optimal_clusters,
        "optimal_score": silhouette_scores[optimal_clusters]
    }
    
    return optimal_clusters, analysis

def create_advanced_models(problem_type: str, input_shape: int = None, 
                          num_classes: int = None, data_size: int = 1000) -> Dict:
    """
    Create a comprehensive set of models based on problem type
    
    Args:
        problem_type: Type of problem (classification, regression, clustering, etc.)
        input_shape: Number of features
        num_classes: Number of classes for classification
        data_size: Size of dataset
        
    Returns:
        Dict: Dictionary of model instances
    """
    models = {}
    
    if problem_type == 'classification':
        # Basic models
        models.update({
            'Random Forest': ensemble.RandomForestClassifier(random_state=42),
            'Gradient Boosting': ensemble.GradientBoostingClassifier(random_state=42),
            'Extra Trees': ensemble.ExtraTreesClassifier(random_state=42),
            'Logistic Regression': linear_model.LogisticRegression(random_state=42),
            'SVM': svm.SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': neighbors.KNeighborsClassifier(),
            'Naive Bayes': naive_bayes.GaussianNB(),
            'Decision Tree': tree.DecisionTreeClassifier(random_state=42),
            'MLP': neural_network.MLPClassifier(random_state=42, max_iter=500)
        })
        
        # Advanced models if available
        if data_size < 10000:  # For smaller datasets
            models['Gaussian Process'] = GaussianProcessClassifier(random_state=42)
            models['QDA'] = QuadraticDiscriminantAnalysis()
        
        # Boosting models
        if XGB_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        if LGB_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = cb.CatBoostClassifier(random_state=42, verbose=False)
    
    elif problem_type == 'regression':
        # Basic models
        models.update({
            'Random Forest': ensemble.RandomForestRegressor(random_state=42),
            'Gradient Boosting': ensemble.GradientBoostingRegressor(random_state=42),
            'Extra Trees': ensemble.ExtraTreesRegressor(random_state=42),
            'Linear Regression': linear_model.LinearRegression(),
            'Ridge': linear_model.Ridge(random_state=42),
            'Lasso': linear_model.Lasso(random_state=42),
            'ElasticNet': linear_model.ElasticNet(random_state=42),
            'SVR': svm.SVR(),
            'K-Nearest Neighbors': neighbors.KNeighborsRegressor(),
            'Decision Tree': tree.DecisionTreeRegressor(random_state=42),
            'MLP': neural_network.MLPRegressor(random_state=42, max_iter=500)
        })
        
        # Advanced models
        if data_size < 10000:
            models['Gaussian Process'] = GaussianProcessRegressor(random_state=42)
        
        # Boosting models
        if XGB_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(random_state=42)
        if LGB_AVAILABLE:
            models['LightGBM'] = lgb.LGBMRegressor(random_state=42, verbose=-1)
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = cb.CatBoostRegressor(random_state=42, verbose=False)
    
    elif problem_type == 'clustering':
        # Basic clustering
        models.update({
            'K-Means': cluster.KMeans(random_state=42),
            'DBSCAN': cluster.DBSCAN(),
            'Agglomerative': cluster.AgglomerativeClustering(),
            'Gaussian Mixture': mixture.GaussianMixture(random_state=42),
            'Mean Shift': MeanShift(),
            'Spectral': SpectralClustering(random_state=42),
            'OPTICS': OPTICS(),
            'Affinity Propagation': AffinityPropagation(random_state=42)
        })
        
        # Advanced clustering
        if HDBSCAN_AVAILABLE:
            models['HDBSCAN'] = hdbscan.HDBSCAN()
    
    elif problem_type == 'anomaly_detection':
        models.update({
            'Isolation Forest': IsolationForest(random_state=42),
            'One-Class SVM': OneClassSVM(),
            'Elliptic Envelope': EllipticEnvelope(random_state=42),
            'Local Outlier Factor': neighbors.LocalOutlierFactor(novelty=True)
        })
    
    return models

def create_advanced_visualizations(df: pd.DataFrame, operation_id: str, 
                                 target_column: str = None) -> List[str]:
    """
    Create comprehensive visualizations for data analysis
    
    Args:
        df: DataFrame to visualize
        operation_id: Operation ID for file naming
        target_column: Target column for supervised learning
        
    Returns:
        List[str]: Paths to generated plot files
    """
    plot_paths = []
    
    try:
        ensure_directories_exist(PLOTS_DIR)
        
        # 1. Data Overview Dashboard
        fig = plt.figure(figsize=(20, 15))
        
        # Create grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Column types distribution
        ax1 = fig.add_subplot(gs[0, 0])
        numeric_count = len(df.select_dtypes(include=['number']).columns)
        categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
        datetime_count = len(df.select_dtypes(include=['datetime']).columns)
        
        types_data = [numeric_count, categorical_count, datetime_count]
        types_labels = ['Numeric', 'Categorical', 'Datetime']
        colors = ['#3498db', '#e74c3c', '#f39c12']
        
        ax1.pie([x for x in types_data if x > 0], 
                labels=[l for l, x in zip(types_labels, types_data) if x > 0],
                autopct='%1.1f%%', startangle=90, colors=colors)
        ax1.set_title('Column Types Distribution', fontsize=14, fontweight='bold')
        
        # Missing values analysis
        ax2 = fig.add_subplot(gs[0, 1:3])
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            missing_df = missing_data[missing_data > 0].sort_values(ascending=False)
            if len(missing_df) > 0:
                ax2.bar(range(len(missing_df)), missing_df.values, color='#e74c3c')
                ax2.set_xticks(range(len(missing_df)))
                ax2.set_xticklabels(missing_df.index, rotation=45, ha='right')
                ax2.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Missing Count')
        else:
            ax2.text(0.5, 0.5, 'No Missing Values!', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=16, color='green',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax2.set_title('Data Quality Status', fontsize=14, fontweight='bold')
            ax2.axis('off')
        
        # Sample distribution
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            ax3 = fig.add_subplot(gs[0, 3])
            sample_col = numeric_cols[0]
            df[sample_col].hist(bins=30, alpha=0.7, ax=ax3, color='#3498db')
            ax3.set_title(f'Distribution: {sample_col}', fontsize=12)
            ax3.set_xlabel(sample_col)
            ax3.set_ylabel('Frequency')
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            ax4 = fig.add_subplot(gs[1, :2])
            correlation_matrix = df[numeric_cols].corr()
            im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax4.set_xticks(range(len(correlation_matrix.columns)))
            ax4.set_yticks(range(len(correlation_matrix.columns)))
            ax4.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
            ax4.set_yticklabels(correlation_matrix.columns)
            ax4.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax4)
        
        # Target variable analysis
        if target_column and target_column in df.columns:
            ax5 = fig.add_subplot(gs[1, 2:])
            if df[target_column].dtype.kind in 'biufc':
                df[target_column].hist(bins=30, alpha=0.7, ax=ax5, color='#2ecc71')
                ax5.set_title(f'Target Distribution: {target_column}', fontsize=14, fontweight='bold')
            else:
                value_counts = df[target_column].value_counts()
                ax5.bar(range(len(value_counts)), value_counts.values, color='#2ecc71')
                ax5.set_xticks(range(len(value_counts)))
                ax5.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax5.set_title(f'Target Distribution: {target_column}', fontsize=14, fontweight='bold')
        
        # Data quality summary
        ax6 = fig.add_subplot(gs[2, :])
        summary_text = f"""
        Dataset Summary:
        • Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns
        • Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
        • Missing Values: {df.isnull().sum().sum():,} ({df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100:.1f}%)
        • Numeric Columns: {len(numeric_cols)}
        • Categorical Columns: {len(df.select_dtypes(include=['object', 'category']).columns)}
        • Duplicate Rows: {df.duplicated().sum():,}
        • Unique Values per Column (avg): {int(df.nunique().mean())}
        """
        ax6.text(0.05, 0.5, summary_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        ax6.axis('off')
        
        plt.suptitle('Advanced Data Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        dashboard_path = os.path.join(PLOTS_DIR, f"{operation_id}_advanced_dashboard.png")
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_paths.append(dashboard_path)
        
        # 2. Advanced dimensionality reduction visualization
        if len(numeric_cols) > 2 and len(df) <= 5000:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Prepare data
            X = df[numeric_cols].fillna(df[numeric_cols].median())
            X_scaled = preprocessing.StandardScaler().fit_transform(X)
            
            # PCA
            pca = decomposition.PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            axes[0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, c='#3498db')
            axes[0].set_title(f'PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.1%})')
            axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            
            # t-SNE
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
                X_tsne = tsne.fit_transform(X_scaled)
                axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, c='#e74c3c')
                axes[1].set_title('t-SNE')
                axes[1].set_xlabel('t-SNE 1')
                axes[1].set_ylabel('t-SNE 2')
            except Exception as e:
                axes[1].text(0.5, 0.5, f't-SNE failed:\n{str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('t-SNE (Failed)')
            
            # UMAP
            if UMAP_AVAILABLE:
                try:
                    umap_reducer = umap.UMAP(n_components=2, random_state=42)
                    X_umap = umap_reducer.fit_transform(X_scaled)
                    axes[2].scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.6, c='#2ecc71')
                    axes[2].set_title('UMAP')
                    axes[2].set_xlabel('UMAP 1')
                    axes[2].set_ylabel('UMAP 2')
                except Exception as e:
                    axes[2].text(0.5, 0.5, f'UMAP failed:\n{str(e)[:50]}...', 
                               ha='center', va='center', transform=axes[2].transAxes)
                    axes[2].set_title('UMAP (Failed)')
            else:
                # Use Isomap as fallback
                try:
                    isomap = Isomap(n_components=2)
                    X_isomap = isomap.fit_transform(X_scaled)
                    axes[2].scatter(X_isomap[:, 0], X_isomap[:, 1], alpha=0.6, c='#f39c12')
                    axes[2].set_title('Isomap')
                    axes[2].set_xlabel('Isomap 1')
                    axes[2].set_ylabel('Isomap 2')
                except:
                    axes[2].text(0.5, 0.5, 'UMAP not available\nIsomap also failed', 
                               ha='center', va='center', transform=axes[2].transAxes)
                    axes[2].set_title('Alternative Method Failed')
            
            plt.suptitle('Advanced Dimensionality Reduction', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            dimred_path = os.path.join(PLOTS_DIR, f"{operation_id}_dimensionality_reduction.png")
            plt.savefig(dimred_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plot_paths.append(dimred_path)
        
        # 3. Interactive visualization with Plotly (if available)
        if PLOTLY_AVAILABLE and len(numeric_cols) >= 2:
            try:
                # Create interactive scatter matrix
                sample_size = min(1000, len(df))
                df_sample = df.sample(n=sample_size, random_state=42)
                
                # Scatter matrix
                fig = px.scatter_matrix(
                    df_sample[numeric_cols[:6]],  # Limit to 6 columns
                    dimensions=numeric_cols[:6],
                    title="Interactive Scatter Matrix",
                    height=800
                )
                
                interactive_path = os.path.join(PLOTS_DIR, f"{operation_id}_interactive_scatter.html")
                fig.write_html(interactive_path)
                plot_paths.append(interactive_path)
                
            except Exception as e:
                logger.warning(f"Failed to create interactive visualization: {e}")
        
    except Exception as e:
        logger.error(f"Error creating advanced visualizations: {e}")
        logger.error(traceback.format_exc())
    
    return plot_paths

def generate_sampling_code(dataset_id: str, sampling_details: Dict, target_column: str = None) -> str:
    """Generate Python code for data sampling"""
    if not sampling_details.get("applied", False):
        return f"# No sampling applied - using full dataset\ndf = pd.read_csv('{dataset_id}.csv')\n"
    
    method = sampling_details["method"]
    sample_size = sampling_details["sampled_rows"]
    random_state = sampling_details.get("random_state", 42)
    
    code = f'''
# Load original dataset
df_original = pd.read_csv('{dataset_id}.csv')
print(f"Original dataset size: {{df_original.shape}}")

# Sampling Strategy: {method}
# Reason: {sampling_details["details"]}
'''
    
    if "stratified" in method:
        code += f'''
# Stratified sampling to preserve target distribution
from sklearn.model_selection import train_test_split

try:
    _, df = train_test_split(
        df_original, 
        test_size={sample_size}, 
        random_state={random_state},
        stratify=df_original['{target_column}']
    )
    print("Applied stratified sampling")
except:
    # Fallback to random sampling if stratified fails
    df = df_original.sample(n={sample_size}, random_state={random_state})
    print("Applied random sampling (stratified fallback)")
'''
    elif "systematic" in method:
        step = sampling_details["original_rows"] // sample_size
        code += f'''
# Systematic sampling for representative coverage
step = {step}
indices = list(range(0, len(df_original), step))[:{sample_size}]
df = df_original.iloc[indices].copy()
print("Applied systematic sampling")
'''
    else:  # random or fallback
        code += f'''
# Random sampling
df = df_original.sample(n={sample_size}, random_state={random_state})
print("Applied random sampling")
'''
    
    code += f'''
print(f"Sampled dataset size: {{df.shape}}")
print(f"Sample ratio: {sampling_details["sample_ratio"]:.3f}")

# Verify sampling quality
print("\\nSampling Quality Check:")
'''
    
    if "quality_check" in sampling_details:
        quality = sampling_details["quality_check"]
        if "overall_quality" in quality:
            code += f'print("Overall quality: {quality["overall_quality"]}")\n'
        if "numeric_similarity" in quality:
            code += f'print("Numeric similarity: {quality["numeric_similarity"]["quality_rating"]}")\n'
        if "target_distribution" in quality:
            code += f'print("Target distribution preservation: {quality["target_distribution"]["quality_rating"]}")\n'
    
    return code

def save_code_as_file(operation_id: str, code_content: str, code_type: str = "analysis") -> str:
    """
    Save Python code as a file and return file path
    
    Args:
        operation_id (str): Operation ID for file naming
        code_content (str): Python code content to save
        code_type (str): Type of code (analysis, visualization, modeling, etc.)
        
    Returns:
        str: Path to saved code file
    """
    try:
        # Ensure code directory exists
        ensure_directories_exist(CODE_DIR)
        
        code_filename = f"{operation_id}_{code_type}.py"
        code_path = os.path.join(CODE_DIR, code_filename)
        
        # Add header comment with metadata
        header = f'''"""
Generated Code for Operation: {operation_id}
Code Type: {code_type}
Generated at: {datetime.now().isoformat()}

This code was automatically generated by the Advanced Data Science MCP Server.
You can run this code independently in your Python environment.
"""

'''
        
        full_content = header + code_content
        
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        logger.info(f"Code saved: {code_path}")
        return code_path
    except Exception as e:
        logger.error(f"Failed to save code: {e}")
        return ""

def generate_eda_code(dataset_id: str, sampling_info: Dict = None) -> str:
    """Generate Python code for EDA operations"""
    code = f'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('{dataset_id}.csv')
'''
    
    # Add sampling information if applicable
    if sampling_info and sampling_info.get("applied", False):
        code += f'''
# NOTE: This dataset has been sampled for performance
# Original size: {sampling_info.get("original_rows", "unknown")} rows
# Current size: {sampling_info.get("sampled_rows", "unknown")} rows
# Sampling method: {sampling_info.get("method", "unknown")}
# Sample quality: {sampling_info.get("quality_check", {}).get("overall_quality", "unknown")}

print("📊 Dataset Information (Sampled Data):")
print(f"Sample size: {{df.shape[0]}} rows (from {sampling_info.get('original_rows', 'unknown')} original)")
print(f"Sampling method: {sampling_info.get('method', 'unknown')}")
print(f"Sample quality: {sampling_info.get('quality_check', {}).get('overall_quality', 'unknown')}")
print("-" * 50)
'''
    
    code += '''
# Basic information
print("Dataset Shape:", df.shape)
print("\\nColumn Information:")
print(df.info())

print("\\nBasic Statistics:")
print(df.describe())

print("\\nMissing Values:")
print(df.isnull().sum())

# Numeric columns analysis
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 1:
    # Correlation analysis
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

# Distribution plots
if len(numeric_cols) > 0:
    plt.figure(figsize=(15, 10))
    df[numeric_cols].hist(bins=30, alpha=0.7)
    plt.suptitle('Distribution of Numeric Variables')
    plt.tight_layout()
    plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

# Missing values visualization
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    plt.figure(figsize=(10, 6))
    missing_values[missing_values > 0].plot(kind='bar')
    plt.title('Missing Values by Column')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('missing_values.png', dpi=300, bbox_inches='tight')
    plt.show()

# Outlier detection
print("\\nOutlier Analysis:")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    if len(outliers) > 0:
        print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

print("\\nEDA completed successfully!")
'''
    
    if sampling_info and sampling_info.get("applied", False):
        code += '''
print("\\n⚠️  IMPORTANT: Results are based on sampled data")
print("   For production decisions, validate with full dataset")
'''
    
    return code

def generate_automl_code(dataset_id: str, target_column: str, problem_type: str, sampling_info: Dict = None) -> str:
    """Generate Python code for AutoML operations"""
    code = f'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Additional advanced models (if available)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False

# Load dataset
df = pd.read_csv('{dataset_id}.csv')
'''
    
    # Add sampling information if applicable
    if sampling_info and sampling_info.get("applied", False):
        code += f'''
# NOTE: This model is trained on sampled data
# Original size: {sampling_info.get("original_rows", "unknown")} rows
# Training size: {sampling_info.get("sampled_rows", "unknown")} rows  
# Sampling method: {sampling_info.get("method", "unknown")}
# Sample quality: {sampling_info.get("quality_check", {}).get("overall_quality", "unknown")}

print("🤖 AutoML Training Information:")
print(f"Training on sampled data: {{df.shape[0]}} rows")
print(f"Original dataset: {sampling_info.get('original_rows', 'unknown')} rows")
print(f"Sampling quality: {sampling_info.get('quality_check', {}).get('overall_quality', 'unknown')}")
print(f"Problem type: {problem_type}")
print("-" * 50)
'''
    
    code += f'''
# Data preprocessing
processed_df = df.copy()

# Handle missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype.kind in 'biufc':  # numeric
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        else:  # categorical
            processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])

# Handle categorical variables
categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
if '{target_column}' in categorical_cols:
    categorical_cols = categorical_cols.drop('{target_column}')

for col in categorical_cols:
    if processed_df[col].nunique() > 10:
        # Frequency encoding for high cardinality
        freq_encoding = processed_df[col].value_counts().to_dict()
        processed_df[col] = processed_df[col].map(freq_encoding)
    else:
        # One-hot encoding for low cardinality
        dummies = pd.get_dummies(processed_df[col], prefix=col)
        processed_df = pd.concat([processed_df.drop(col, axis=1), dummies], axis=1)

# Prepare features and target
X = processed_df.drop('{target_column}', axis=1)
y = processed_df['{target_column}']

# Handle target encoding for classification
label_encoder = None
if '{problem_type}' == 'classification' and y.dtype == 'object':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42,
    stratify=y if '{problem_type}' == 'classification' else None
)

# Model selection
models = {{}}
if '{problem_type}' == 'classification':
    models.update({{
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(random_state=42)
    }})
    if XGB_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    if LGB_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
else:  # regression
    models.update({{
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'SVR': SVR()
    }})
    if XGB_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(random_state=42)
    if LGB_AVAILABLE:
        models['LightGBM'] = lgb.LGBMRegressor(random_state=42, verbose=-1)

# Train and evaluate models
model_scores = {{}}
best_score = {'np.inf' if problem_type == 'regression' else '0'}
best_model = None
best_model_name = None

for name, model in models.items():
    print(f"Training {{name}}...")
    
    # Cross-validation
    scoring = 'accuracy' if '{problem_type}' == 'classification' else 'neg_mean_squared_error'
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    if '{problem_type}' == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        model_scores[name] = {{
            'cv_score': cv_scores.mean(),
            'test_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }}
        print(f"{{name}} - Accuracy: {{accuracy:.4f}}, F1: {{f1:.4f}}")
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_model_name = name
    else:  # regression
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        model_scores[name] = {{
            'cv_score': -cv_scores.mean(),
            'test_r2': r2,
            'rmse': rmse,
            'mae': mae
        }}
        print(f"{{name}} - R2: {{r2:.4f}}, RMSE: {{rmse:.4f}}")
        
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name

print(f"\\nBest model: {{best_model_name}}")
print(f"Best model scores: {{model_scores[best_model_name]}}")

# Save the best model
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
if label_encoder:
    joblib.dump(label_encoder, 'label_encoder.pkl')

print("\\nModels saved successfully!")

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({{
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }}).sort_values('importance', ascending=False)
    
    print("\\nTop 10 Feature Importances:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.title(f'Top 10 Feature Importances - {{best_model_name}}')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
'''
    
    if sampling_info and sampling_info.get("applied", False):
        code += '''
print("\\n⚠️  IMPORTANT NOTES:")
print("   - Model trained on sampled data for performance")
print("   - Validate model performance on full dataset before production use")
print("   - Consider retraining with full data for final model")
'''
    
    return code

def generate_visualization_code(dataset_id: str, plot_type: str, x_column: str = None, y_column: str = None, title: str = None) -> str:
    """Generate Python code for visualization operations"""
    code = f'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('{dataset_id}.csv')

# Create visualization
plt.figure(figsize=(10, 6))

'''
    
    if plot_type.lower() == 'histogram':
        code += f'''
# Histogram plot
df['{x_column}'].hist(bins=30, alpha=0.7)
plt.xlabel('{x_column}')
plt.ylabel('Frequency')
'''
    elif plot_type.lower() == 'scatter':
        code += f'''
# Scatter plot
plt.scatter(df['{x_column}'], df['{y_column}'], alpha=0.6)
plt.xlabel('{x_column}')
plt.ylabel('{y_column}')
'''
    elif plot_type.lower() == 'boxplot':
        code += f'''
# Box plot
df.boxplot(column='{x_column}')
'''
    elif plot_type.lower() == 'heatmap':
        code += f'''
# Correlation heatmap
numeric_cols = df.select_dtypes(include=['number'])
correlation_matrix = numeric_cols.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
'''
    elif plot_type.lower() == 'bar':
        if y_column:
            code += f'''
# Bar plot with aggregation
df.groupby('{x_column}')['{y_column}'].mean().plot(kind='bar')
plt.ylabel('Mean {y_column}')
'''
        else:
            code += f'''
# Bar plot of value counts
df['{x_column}'].value_counts().plot(kind='bar')
plt.ylabel('Count')
'''
        code += f'''
plt.xlabel('{x_column}')
plt.xticks(rotation=45)
'''
    elif plot_type.lower() == 'pairplot':
        code += f'''
# Pair plot for multiple variables
numeric_cols = df.select_dtypes(include=['number']).columns[:6]  # Limit to 6 columns
sns.pairplot(df[numeric_cols])
'''
    elif plot_type.lower() == 'violin':
        code += f'''
# Violin plot
if df['{y_column}'].dtype.kind in 'biufc':
    sns.violinplot(x='{x_column}', y='{y_column}', data=df)
else:
    sns.violinplot(x='{y_column}', y='{x_column}', data=df)
plt.xticks(rotation=45)
'''
    
    code += f'''
plt.title('{title or f"{plot_type.title()} Plot"}')
plt.tight_layout()
plt.savefig('{plot_type}_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization completed successfully!")
'''
    return code

def safe_dataframe_summary(df: pd.DataFrame, max_rows: int = 100, sampling_info: Dict = None) -> Dict:
    """
    Generate a safe summary of DataFrame without sending large data to LLM
    """
    # Convert pandas dtypes to JSON-serializable strings
    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Convert missing values to regular Python integers
    missing_values = {col: int(count) for col, count in df.isnull().sum().items()}
    
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': dtypes_dict,
        'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        'missing_values': missing_values,
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
    }
    
    # Add sampling information if available
    if sampling_info and sampling_info.get("applied", False):
        summary['sampling_info'] = {
            'is_sampled': True,
            'original_rows': sampling_info.get('original_rows', 'unknown'),
            'current_rows': df.shape[0],
            'sampling_method': sampling_info.get('method', 'unknown'),
            'sample_ratio': sampling_info.get('sample_ratio', 0),
            'sample_quality': sampling_info.get('quality_check', {}).get('overall_quality', 'unknown'),
            'reduction_percentage': f"{(1 - sampling_info.get('sample_ratio', 1)) * 100:.1f}%"
        }
    else:
        summary['sampling_info'] = {
            'is_sampled': False,
            'note': 'Full dataset used for analysis'
        }
    
    # Add sample data only if dataset is small
    if len(df) <= max_rows:
        # Use safe JSON serialization
        summary['head'] = safe_json_serialize(df.head())
        summary['describe'] = safe_json_serialize(df.describe())
    else:
        # Use safe JSON serialization for samples
        summary['head'] = safe_json_serialize(df.head(5))
        sample_describe = df.sample(min(1000, len(df))).describe()
        summary['describe_sample'] = safe_json_serialize(sample_describe)
        summary['note'] = f"Large dataset ({len(df)} rows). Showing sample statistics."
    
    return summary

def get_dataset_sampling_info(dataset_id: str) -> Dict:
    """
    Retrieve sampling information for a dataset from tracker operations
    """
    # Look for the load operation that created this dataset
    for op_id, operation in tracker.operations.items():
        if (operation.get('type') == 'load_dataset' and 
            operation.get('output_data', {}).get('dataset_id') == dataset_id):
            sampling_info = operation.get('output_data', {}).get('sampling_info', {})
            if sampling_info.get('details', {}).get('applied', False):
                return sampling_info.get('details', {})
    
    return {"applied": False}

def analyze_supervised_learning_potential(df: pd.DataFrame) -> Dict:
    """
    Analyze dataset for supervised learning potential and suggest target columns.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dict: Analysis results with recommended target columns
    """
    analysis = {
        "has_potential": False,
        "recommended_targets": [],
        "analysis_details": {}
    }
    
    try:
        # Analyze each column for target potential
        target_candidates = []
        
        for col in df.columns:
            col_analysis = {
                "column": col,
                "dtype": str(df[col].dtype),
                "unique_count": df[col].nunique(),
                "missing_ratio": df[col].isnull().sum() / len(df),
                "suitability_score": 0,
                "suggested_type": None,
                "reasons": []
            }
            
            # Skip if too many missing values
            if col_analysis["missing_ratio"] > 0.5:
                col_analysis["reasons"].append("Too many missing values (>50%)")
                continue
            
            # Skip if only one unique value
            if col_analysis["unique_count"] <= 1:
                col_analysis["reasons"].append("Only one unique value")
                continue
            
            # Check for common target column names
            target_keywords = ['target', 'label', 'class', 'category', 'outcome', 
                             'result', 'y', 'survived', 'churn', 'fraud', 'default']
            if any(keyword in col.lower() for keyword in target_keywords):
                col_analysis["suitability_score"] += 30
                col_analysis["reasons"].append("Column name suggests target variable")
            
            # Analyze for classification potential
            if df[col].dtype.kind in 'O':  # Object/categorical
                if col_analysis["unique_count"] <= 20:  # Reasonable number of classes
                    col_analysis["suitability_score"] += 70
                    col_analysis["suggested_type"] = "classification"
                    col_analysis["reasons"].append(f"Categorical with {col_analysis['unique_count']} classes")
                elif col_analysis["unique_count"] <= 50:
                    col_analysis["suitability_score"] += 40
                    col_analysis["suggested_type"] = "classification"
                    col_analysis["reasons"].append(f"High cardinality categorical ({col_analysis['unique_count']} classes)")
            
            # Analyze for regression potential
            elif df[col].dtype.kind in 'biufc':  # Numeric
                unique_ratio = col_analysis["unique_count"] / len(df)
                
                if unique_ratio > 0.1:  # Continuous-like
                    col_analysis["suitability_score"] += 80
                    col_analysis["suggested_type"] = "regression"
                    col_analysis["reasons"].append("Continuous numeric variable")
                elif col_analysis["unique_count"] <= 20:  # Discrete classes
                    col_analysis["suitability_score"] += 60
                    col_analysis["suggested_type"] = "classification"
                    col_analysis["reasons"].append(f"Discrete numeric with {col_analysis['unique_count']} values")
            
            if col_analysis["suitability_score"] > 0:
                target_candidates.append(col_analysis)
        
        # Sort by suitability score
        target_candidates.sort(key=lambda x: x["suitability_score"], reverse=True)
        
        analysis["recommended_targets"] = target_candidates[:5]  # Top 5 candidates
        analysis["has_potential"] = len(target_candidates) > 0
        analysis["analysis_details"] = {
            "total_columns": len(df.columns),
            "candidates_found": len(target_candidates),
            "best_candidate": target_candidates[0] if target_candidates else None
        }
        
    except Exception as e:
        analysis["error"] = f"Analysis failed: {str(e)}"
    
    return analysis

def create_comprehensive_report(
    operations: List[str],
    report_title: str = "Advanced Data Science Analysis Report",
    dataset_name: str = "Dataset",
    *,
    embed_figures: bool = True,
    add_table_of_contents: bool = True,
    use_llm: bool = True,
) -> str:
    """
    Create a comprehensive markdown report from multiple operations
    Enhanced version with detailed code analysis and practical usage examples
    """
    
    # Generate report ID
    report_id = f"report_{uuid.uuid4().hex[:8]}"
    
    # ════════════════════════════════════════════════════════════════════════
    # Helper Functions
    # ════════════════════════════════════════════════════════════════════════
    
    def _rel(p: str) -> str:
        """Convert absolute path to relative path from reports directory"""
        try:
            return os.path.relpath(p, start=REPORTS_DIR).replace("\\", "/")
        except Exception:
            return os.path.basename(p)
    
    def _format_number(value, decimals=2, default="N/A"):
        """Safely format numeric values"""
        if value is None:
            return default
        try:
            if isinstance(value, (int, np.integer)):
                return f"{int(value):,}"
            else:
                return f"{float(value):,.{decimals}f}"
        except:
            return default
    
    def _format_percentage(value, decimals=1, default="N/A"):
        """Safely format percentage values"""
        if value is None:
            return default
        try:
            return f"{float(value) * 100:.{decimals}f}%"
        except:
            return default
    
    def _load_all_operations() -> Dict[str, Dict]:
        """Load all operations from logs directory"""
        all_operations = {}
        
        # First, load from tracker memory
        for op_id, op_data in tracker.operations.items():
            all_operations[op_id] = op_data
        
        # Then, load from log files
        if os.path.exists(LOGS_DIR):
            for log_file in glob.glob(os.path.join(LOGS_DIR, "operation_*.json")):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        op_data = json.load(f)
                        op_id = op_data.get('id', os.path.basename(log_file).replace('operation_', '').replace('.json', ''))
                        all_operations[op_id] = op_data
                except Exception as e:
                    logger.warning(f"Failed to load {log_file}: {e}")
        
        return all_operations
    
    def _extract_code_info(code_path: str) -> Dict:
        """Extract detailed information from Python code file"""
        try:
            with open(code_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split('.')[0])
            
            # Extract key operations and outputs
            key_lines = []
            outputs = []
            
            lines = content.splitlines()
            for i, line in enumerate(lines):
                line_strip = line.strip()
                
                # Skip comments and empty lines
                if line_strip.startswith('#') or not line_strip:
                    continue
                
                # Extract key operations
                key_operations = [
                    'corr()', 'describe()', 'hist(', 'heatmap', 'boxplot', 'scatter',
                    'savefig', 'dump', 'fit(', 'transform(', 'predict(', 'score(',
                    'train_test_split', 'StandardScaler', 'fillna', 'get_dummies',
                    'value_counts', 'groupby', 'plot', 'cross_val_score'
                ]
                
                if any(op in line for op in key_operations):
                    if len(key_lines) < 10:  # Limit to 10 key lines
                        key_lines.append(line_strip)
                
                # Extract outputs
                if 'savefig' in line:
                    # Extract filename
                    if "'" in line:
                        parts = line.split("'")
                        if len(parts) >= 2:
                            filename = parts[1]
                            outputs.append(f"시각화 파일: {filename}")
                    elif '"' in line:
                        parts = line.split('"')
                        if len(parts) >= 2:
                            filename = parts[1]
                            outputs.append(f"시각화 파일: {filename}")
                elif 'to_csv' in line:
                    outputs.append("데이터 파일: CSV 형식")
                elif 'joblib.dump' in line or '.pkl' in line:
                    if "'" in line and '.pkl' in line:
                        parts = line.split("'")
                        for part in parts:
                            if '.pkl' in part:
                                outputs.append(f"모델 파일: {part}")
                                break
                    elif '"' in line and '.pkl' in line:
                        parts = line.split('"')
                        for part in parts:
                            if '.pkl' in part:
                                outputs.append(f"모델 파일: {part}")
                                break
            
            # Determine code type and features
            filename = os.path.basename(code_path)
            
            features = []
            description = ""
            
            if 'eda' in filename.lower():
                description = "탐색적 데이터 분석 코드로 데이터의 기본 통계, 상관관계, 분포, 결측값 패턴을 분석합니다."
                features = [
                    "기술통계 계산 (df.describe())",
                    "상관관계 히트맵 시각화",
                    "변수 분포 히스토그램",
                    "결측값 패턴 분석 및 시각화"
                ]
            elif 'automl' in filename.lower():
                description = "자동화된 머신러닝 파이프라인으로 전처리부터 모델 학습, 평가까지 수행합니다."
                features = [
                    "결측값 자동 처리 (중앙값/최빈값)",
                    "범주형 변수 인코딩 (원-핫/빈도 인코딩)",
                    "특성 스케일링 (StandardScaler)",
                    "다중 모델 비교 및 성능 평가",
                    "최적 모델 자동 선택 및 저장"
                ]
            elif 'data_loading' in filename.lower() or 'load' in filename.lower():
                description = "데이터 로딩 및 샘플링 전략을 포함한 데이터 준비 코드입니다."
                features = [
                    "대용량 파일 지능형 샘플링",
                    "데이터 품질 검증",
                    "메모리 효율적 로딩"
                ]
            else:
                description = f"{filename.split('_')[1] if '_' in filename else 'analysis'} 관련 데이터 분석 코드입니다."
            
            return {
                'imports': list(set(imports)),
                'key_lines': key_lines,
                'outputs': outputs,
                'features': features,
                'description': description,
                'lines': len(content.splitlines()),
                'size_kb': len(content) / 1024,
                'filename': filename
            }
        except Exception as e:
            logger.warning(f"Failed to analyze code {code_path}: {e}")
            return {}
    
    # ════════════════════════════════════════════════════════════════════════
    # Data Collection
    # ════════════════════════════════════════════════════════════════════════
    
    logger.info("Starting comprehensive report generation")
    
    # Load all operations
    all_operations = _load_all_operations()
    
    # Initialize data containers
    dataset_info = None
    sampling_info = None
    eda_results = None
    automl_results = None
    viz_results = []
    all_artifacts = []
    
    # Process requested operations
    for op_id in operations:
        if op_id in all_operations:
            op = all_operations[op_id]
            op_type = op.get('type', '')
            
            logger.info(f"Processing operation {op_id} of type {op_type}")
            
            if op_type == 'load_dataset':
                dataset_info = op.get('output', {}) or op.get('output_data', {})
                sampling_info = dataset_info.get('sampling_info', {})
            elif op_type == 'perform_eda':
                eda_results = op.get('output', {}) or op.get('output_data', {})
            elif op_type == 'auto_ml_pipeline':
                automl_results = op.get('output', {}) or op.get('output_data', {})
            elif op_type == 'create_visualization':
                viz_results.append(op.get('output', {}) or op.get('output_data', {}))
            
            # Collect artifacts
            files = op.get('files_created', [])
            all_artifacts.extend(files)
    
    # Collect all artifacts from directories
    artifact_patterns = [
        (PLOTS_DIR, "*.png", "plots"),
        (CODE_DIR, "*.py", "code"),
        (MODELS_DIR, "*.pkl", "models"),
        (DATASETS_DIR, "*.csv", "datasets"),
        (REPORTS_DIR, "*.json", "reports")
    ]
    
    categorized_artifacts = {
        'plots': [],
        'code': [],
        'models': [],
        'datasets': [],
        'reports': []
    }
    
    for directory, pattern, category in artifact_patterns:
        if os.path.exists(directory):
            for filepath in glob.glob(os.path.join(directory, pattern)):
                if filepath not in all_artifacts:
                    all_artifacts.append(filepath)
                categorized_artifacts[category].append(filepath)
    
    # ════════════════════════════════════════════════════════════════════════
    # Report Generation
    # ════════════════════════════════════════════════════════════════════════
    
    report_lines = []
    
    # Header
    report_lines.extend([
        f"# {report_title}",
        "",
        f"**데이터셋**: {dataset_name}  ",
        f"**분석 일시**: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}  ",
        f"**보고서 ID**: `{report_id}`",
        "",
        "---",
        ""
    ])
    
    # Executive Summary
    report_lines.extend([
        "## 📋 Executive Summary",
        ""
    ])
    
    # Generate executive summary based on analysis results
    if automl_results and automl_results.get("best_model"):
        best_model = automl_results["best_model"]
        model_name = best_model.get("name", "N/A")
        problem_type = automl_results.get("problem_type", "unknown")
        
        if problem_type == "classification":
            scores = best_model.get("scores", {})
            accuracy = scores.get("test_accuracy", 0)
            f1 = scores.get("f1_score", 0)
            report_lines.extend([
                f"본 분석에서는 **{dataset_name}** 데이터셋에 대한 포괄적인 탐색적 데이터 분석 및 머신러닝 모델링을 수행했습니다. ",
                f"최종적으로 **{model_name}** 모델이 최고 성능을 보였으며, 테스트 정확도 **{accuracy:.1%}**, F1 점수 **{f1:.3f}**를 달성했습니다.",
                ""
            ])
        elif problem_type == "regression":
            scores = best_model.get("scores", {})
            r2 = scores.get("test_r2", 0)
            rmse = scores.get("rmse", 0)
            report_lines.extend([
                f"본 분석에서는 **{dataset_name}** 데이터셋에 대한 포괄적인 탐색적 데이터 분석 및 머신러닝 모델링을 수행했습니다. ",
                f"최종적으로 **{model_name}** 모델이 최고 성능을 보였으며, R² Score **{r2:.3f}**, RMSE **{rmse:.3f}**를 달성했습니다.",
                ""
            ])
        else:
            report_lines.extend([
                f"본 분석에서는 **{dataset_name}** 데이터셋에 대한 포괄적인 데이터 분석을 수행했습니다.",
                ""
            ])
    else:
        report_lines.extend([
            f"본 분석에서는 **{dataset_name}** 데이터셋에 대한 포괄적인 탐색적 데이터 분석을 수행했습니다.",
            ""
        ])
    
    report_lines.extend([
        "---",
        ""
    ])
    
    # Table of Contents
    if add_table_of_contents:
        report_lines.extend([
            "## 목차",
            "",
            "1. [데이터셋 개요](#1-데이터셋-개요)",
            "2. [탐색적 데이터 분석 (EDA)](#2-탐색적-데이터-분석-eda)",
            "3. [머신러닝 모델링](#3-머신러닝-모델링)",
            "4. [시각화 결과](#4-시각화-결과)",
            "5. [생성된 코드](#5-생성된-코드)",
            "6. [산출물 활용 가이드](#6-산출물-활용-가이드)",
            "7. [결론 및 권장사항](#7-결론-및-권장사항)",
            "",
            "---",
            ""
        ])
    
    # ════════════════════════════════════════════════════════════════════════
    # 1. Dataset Information
    # ════════════════════════════════════════════════════════════════════════
    
    report_lines.extend([
        "## 1. 데이터셋 개요",
        ""
    ])
    
    if dataset_info:
        file_info = dataset_info.get('file_info', {})
        summary = dataset_info.get('summary', {})
        
        # 1.1 Basic Information
        report_lines.extend([
            "### 1.1 데이터 기본 정보",
            "",
            f"- **원본 파일**: `{file_info.get('original_file', 'N/A')}`",
            f"- **파일 크기**: {_format_number(file_info.get('file_size_mb', 0), 2)} MB",
            f"- **데이터 규모**: {_format_number(file_info.get('original_rows', 0))} 행 × {_format_number(file_info.get('original_columns', 0))} 열",
            f"- **메모리 사용량**: {_format_number(summary.get('memory_usage_mb', 0), 2)} MB",
            ""
        ])
        
        # 1.2 Variable Composition
        if summary.get('columns'):
            numeric_cols = summary.get('numeric_columns', [])
            categorical_cols = summary.get('categorical_columns', [])
            
            report_lines.extend([
                "### 1.2 변수 구성",
                "",
                f"**수치형 변수 ({len(numeric_cols)}개)**: `{', '.join(numeric_cols)}`",
                "",
                f"**범주형 변수 ({len(categorical_cols)}개)**: `{', '.join(categorical_cols)}`",
                ""
            ])
        
        # 1.3 Data Quality
        missing_values = summary.get('missing_values', {})
        if missing_values:
            total_missing = sum(v for v in missing_values.values() if v is not None)
            total_cells = file_info.get('original_rows', 1) * file_info.get('original_columns', 1)
            missing_ratio = (total_missing / total_cells * 100) if total_cells > 0 else 0
            
            report_lines.extend([
                "### 1.3 데이터 품질",
                "",
                f"- **전체 결측값**: {_format_number(total_missing)}개 ({missing_ratio:.1f}%)",
                "- **주요 결측 변수**:",
                ""
            ])
            
            # Show top missing variables
            sorted_missing = sorted(
                [(col, count) for col, count in missing_values.items() if count is not None and count > 0],
                key=lambda x: x[1], reverse=True
            )
            
            for col, count in sorted_missing[:5]:
                col_ratio = (count / file_info.get('original_rows', 1) * 100) if file_info.get('original_rows', 1) > 0 else 0
                report_lines.append(f"  - `{col}`: {_format_number(count)}개 ({col_ratio:.1f}%)")
            
            report_lines.append("")
        
        # 1.4 Data Files
        if categorized_artifacts['datasets']:
            report_lines.extend([
                "### 1.4 데이터 파일",
                "",
                "| 파일명 | 크기 (MB) | 설명 |",
                "|--------|-----------|------|"
            ])
            
            for dataset_path in categorized_artifacts['datasets']:
                filename = os.path.basename(dataset_path)
                size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
                
                if 'dataset_' in filename:
                    desc = "처리된 데이터"
                else:
                    desc = "원본 데이터"
                
                report_lines.append(f"| `{filename}` | {size_mb:.2f} | {desc} |")
            
            report_lines.append("")
    
    # ════════════════════════════════════════════════════════════════════════
    # 2. EDA Results
    # ════════════════════════════════════════════════════════════════════════
    
    if eda_results:
        report_lines.extend([
            "## 2. 탐색적 데이터 분석 (EDA)",
            ""
        ])
        
        analysis = eda_results.get('analysis', {})
        
        # 2.1 Statistical Summary
        basic_info = analysis.get('basic_info', {})
        if basic_info.get('describe') or basic_info.get('describe_sample'):
            report_lines.extend([
                "### 2.1 기술통계 요약",
                "",
                "#### 수치형 변수 통계",
                "",
                "| 변수 | 평균 | 표준편차 | 최솟값 | 중앙값 | 최댓값 | 결측값 |",
                "|------|------|----------|--------|--------|--------|--------|"
            ])
            
            describe_data = basic_info.get('describe') or basic_info.get('describe_sample', {})
            missing_values = basic_info.get('missing_values', {})
            
            for var in describe_data.keys():
                if var in describe_data and isinstance(describe_data[var], dict):
                    stats = describe_data[var]
                    missing_count = missing_values.get(var, 0)
                    
                    mean_val = _format_number(stats.get('mean'), 2)
                    std_val = _format_number(stats.get('std'), 2)
                    min_val = _format_number(stats.get('min'), 2)
                    median_val = _format_number(stats.get('50%'), 2)
                    max_val = _format_number(stats.get('max'), 2)
                    
                    report_lines.append(
                        f"| `{var}` | {mean_val} | {std_val} | {min_val} | {median_val} | {max_val} | {_format_number(missing_count)} |"
                    )
            
            report_lines.append("")
        
        # 2.2 Correlation Analysis
        correlation_data = analysis.get('correlation', {})
        strong_correlations = analysis.get('strong_correlations', [])
        
        if strong_correlations:
            report_lines.extend([
                "### 2.2 변수 간 상관관계 분석",
                "",
                "**주요 상관관계 (|r| > 0.3)**:",
                ""
            ])
            
            for corr in strong_correlations[:10]:
                var1 = corr.get('var1', 'N/A')
                var2 = corr.get('var2', 'N/A')
                corr_val = corr.get('correlation', 0)
                strength = corr.get('strength', '')
                
                direction = "양의" if corr_val > 0 else "음의"
                report_lines.append(f"- `{var1}` ↔ `{var2}`: {corr_val:.3f} ({strength} {direction} 상관관계)")
            
            report_lines.append("")
        
        # 2.3 Key Findings
        insights = eda_results.get('insights', [])
        if insights:
            report_lines.extend([
                "### 2.3 주요 발견사항",
                ""
            ])
            
            for insight in insights:
                report_lines.append(f"- {insight}")
            
            report_lines.append("")
    
    # ════════════════════════════════════════════════════════════════════════
    # 3. Machine Learning Results
    # ════════════════════════════════════════════════════════════════════════
    
    if automl_results:
        report_lines.extend([
            "## 3. 머신러닝 모델링",
            ""
        ])
        
        # 3.1 Problem Definition
        problem_type = automl_results.get('problem_type', 'N/A')
        target_column = automl_results.get('target_column', 'N/A')
        
        report_lines.extend([
            "### 3.1 문제 정의 및 접근법",
            "",
            f"- **문제 유형**: {problem_type.title()}",
            f"- **타겟 변수**: `{target_column}`",
            ""
        ])
        
        # 3.2 Preprocessing
        preprocessing_steps = automl_results.get('preprocessing_steps', [])
        if preprocessing_steps:
            report_lines.extend([
                "### 3.2 데이터 전처리",
                "",
                "**적용된 전처리 기법**:",
                ""
            ])
            
            # Group preprocessing steps
            missing_handling = [step for step in preprocessing_steps if 'filled with' in step]
            encoding_steps = [step for step in preprocessing_steps if 'encoding' in step]
            
            if missing_handling:
                report_lines.extend([
                    "**결측값 처리**:",
                    ""
                ])
                for i, step in enumerate(missing_handling[:5]):
                    report_lines.append(f"- {step}")
                if len(missing_handling) > 5:
                    report_lines.append(f"- ... 및 {len(missing_handling) - 5}개 추가 변수")
                report_lines.append("")
            
            if encoding_steps:
                report_lines.extend([
                    "**범주형 변수 인코딩**:",
                    ""
                ])
                for step in encoding_steps:
                    report_lines.append(f"- {step}")
                report_lines.append("")
        
        # 3.3 Model Performance
        model_results = automl_results.get('model_results', {})
        if model_results:
            report_lines.extend([
                "### 3.3 모델 성능 비교",
                ""
            ])
            
            if problem_type.lower() == 'classification':
                report_lines.extend([
                    "| 모델 | 교차검증 점수 | 테스트 정확도 | Precision | Recall | F1 Score |",
                    "|------|---------------|---------------|-----------|--------|----------|"
                ])
                
                for model_name, scores in model_results.items():
                    cv_score = _format_number(scores.get('cv_score', 0), 4)
                    accuracy = _format_number(scores.get('test_accuracy', 0), 4)
                    precision = _format_number(scores.get('precision', 0), 4)
                    recall = _format_number(scores.get('recall', 0), 4)
                    f1 = _format_number(scores.get('f1_score', 0), 4)
                    
                    report_lines.append(
                        f"| {model_name} | {cv_score} | {accuracy} | {precision} | {recall} | {f1} |"
                    )
            else:  # regression
                report_lines.extend([
                    "| 모델 | 교차검증 점수 | R² Score | RMSE | MAE |",
                    "|------|---------------|----------|------|-----|"
                ])
                
                for model_name, scores in model_results.items():
                    cv_score = _format_number(scores.get('cv_score', 0), 4)
                    r2 = _format_number(scores.get('test_r2', 0), 4)
                    rmse = _format_number(scores.get('rmse', 0), 4)
                    mae = _format_number(scores.get('mae', 0), 4)
                    
                    report_lines.append(
                        f"| {model_name} | {cv_score} | {r2} | {rmse} | {mae} |"
                    )
            
            report_lines.append("")
        
        # 3.4 Best Model
        best_model = automl_results.get('best_model', {})
        if best_model:
            model_name = best_model.get('name', 'N/A')
            scores = best_model.get('scores', {})
            
            report_lines.extend([
                f"### 3.4 최적 모델: {model_name}",
                ""
            ])
            
            if problem_type.lower() == 'classification':
                report_lines.extend([
                    f"**{model_name}** 모델이 최고 성능을 보였으며, 주요 성능 지표는 다음과 같습니다:",
                    "",
                    f"- **테스트 정확도**: {_format_percentage(scores.get('test_accuracy', 0))}",
                    f"- **F1 Score**: {_format_number(scores.get('f1_score', 0), 4)}",
                    f"- **교차검증 점수**: {_format_number(scores.get('cv_score', 0), 4)} ± {_format_number(scores.get('cv_std', 0), 4)}",
                    ""
                ])
            else:  # regression
                report_lines.extend([
                    f"**{model_name}** 모델이 최고 성능을 보였으며, 주요 성능 지표는 다음과 같습니다:",
                    "",
                    f"- **R² Score**: {_format_number(scores.get('test_r2', 0), 4)}",
                    f"- **RMSE**: {_format_number(scores.get('rmse', 0), 4)}",
                    f"- **MAE**: {_format_number(scores.get('mae', 0), 4)}",
                    ""
                ])
        
        # 3.5 Feature Importance
        feature_importance = automl_results.get('feature_importance', {})
        if feature_importance and isinstance(feature_importance, dict):
            features = feature_importance.get('feature', {})
            importances = feature_importance.get('importance', {})
            
            if features and importances:
                report_lines.extend([
                    "### 3.5 특성 중요도 분석",
                    "",
                    "**상위 10개 중요 특성**:",
                    "",
                    "| 순위 | 특성 | 중요도 | 기여도 |",
                    "|------|------|--------|--------|"
                ])
                
                # Convert to list and sort
                feature_list = []
                for idx, feature in features.items():
                    if idx in importances:
                        feature_list.append((feature, importances[idx]))
                
                feature_list.sort(key=lambda x: x[1], reverse=True)
                
                total_importance = sum([imp for _, imp in feature_list])
                for rank, (feature, importance) in enumerate(feature_list[:10], 1):
                    contribution = (importance / total_importance * 100) if total_importance > 0 else 0
                    report_lines.append(
                        f"| {rank} | `{feature}` | {_format_number(importance, 4)} | {contribution:.1f}% |"
                    )
                
                report_lines.append("")
    
# ════════════════════════════════════════════════════════════════════════
    # 4. Visualizations
    # ════════════════════════════════════════════════════════════════════════
    
    if categorized_artifacts['plots']:
        report_lines.extend([
            "## 4. 시각화 결과",
            ""
        ])
        
        # Group plots by type with more comprehensive matching
        plot_groups = {
            'dashboard': [],
            'dimensionality': [],
            'correlation': [],
            'distribution': [],
            'missing': [],
            'feature': [],
            'clustering': [],
            'other': []
        }
        
        for plot_path in categorized_artifacts['plots']:
            filename = os.path.basename(plot_path).lower()
            
            # More comprehensive categorization
            if 'dashboard' in filename:
                plot_groups['dashboard'].append(plot_path)
            elif 'dimensionality' in filename or 'dimension' in filename or 'pca' in filename or 'tsne' in filename or 'umap' in filename:
                plot_groups['dimensionality'].append(plot_path)
            elif 'correlation' in filename or 'corr' in filename or 'heatmap' in filename:
                plot_groups['correlation'].append(plot_path)
            elif 'distribution' in filename or 'dist' in filename or 'histogram' in filename:
                plot_groups['distribution'].append(plot_path)
            elif 'missing' in filename:
                plot_groups['missing'].append(plot_path)
            elif 'feature' in filename or 'importance' in filename:
                plot_groups['feature'].append(plot_path)
            elif 'cluster' in filename:
                plot_groups['clustering'].append(plot_path)
            else:
                plot_groups['other'].append(plot_path)
        
        # Display plots in logical order
        section_num = 1
        
        # Dashboard plots
        if plot_groups['dashboard']:
            report_lines.extend([
                f"### 4.{section_num} 데이터 분석 대시보드",
                ""
            ])
            for i, plot_path in enumerate(plot_groups['dashboard']):
                if i > 0:
                    report_lines.append("")
                report_lines.extend([
                    f"![데이터 분석 대시보드 {i+1}]({_rel(plot_path)})",
                    "",
                    "포괄적인 데이터 분석 대시보드로 데이터의 전반적인 특성, 결측값 패턴, 상관관계 등을 한눈에 볼 수 있습니다.",
                    ""
                ])
            section_num += 1
        
        # Dimensionality reduction plots
        if plot_groups['dimensionality']:
            report_lines.extend([
                f"### 4.{section_num} 차원 축소 분석",
                ""
            ])
            for i, plot_path in enumerate(plot_groups['dimensionality']):
                if i > 0:
                    report_lines.append("")
                report_lines.extend([
                    f"![차원 축소 시각화 {i+1}]({_rel(plot_path)})",
                    "",
                    "PCA, t-SNE, UMAP 등의 고급 차원 축소 기법을 사용하여 고차원 데이터를 2차원으로 시각화했습니다. 데이터의 구조와 패턴을 파악할 수 있습니다.",
                    ""
                ])
            section_num += 1
        
        # Correlation plots
        if plot_groups['correlation']:
            report_lines.extend([
                f"### 4.{section_num} 상관관계 분석",
                ""
            ])
            for i, plot_path in enumerate(plot_groups['correlation']):
                if i > 0:
                    report_lines.append("")
                report_lines.extend([
                    f"![상관관계 히트맵 {i+1}]({_rel(plot_path)})",
                    "",
                    "변수 간 상관관계를 시각화한 히트맵입니다. 강한 상관관계(절댓값 0.7 이상)는 모델 성능에 중요한 영향을 미칠 수 있습니다.",
                    ""
                ])
            section_num += 1
        
        # Distribution plots
        if plot_groups['distribution']:
            report_lines.extend([
                f"### 4.{section_num} 변수 분포 분석",
                ""
            ])
            for i, plot_path in enumerate(plot_groups['distribution']):
                if i > 0:
                    report_lines.append("")
                report_lines.extend([
                    f"![변수 분포도 {i+1}]({_rel(plot_path)})",
                    "",
                    "각 수치형 변수의 분포를 히스토그램으로 나타낸 것입니다. 분포의 치우침, 이상치, 다중모드 등을 확인할 수 있습니다.",
                    ""
                ])
            section_num += 1
        
        # Missing value plots
        if plot_groups['missing']:
            report_lines.extend([
                f"### 4.{section_num} 결측값 패턴 분석",
                ""
            ])
            for i, plot_path in enumerate(plot_groups['missing']):
                if i > 0:
                    report_lines.append("")
                report_lines.extend([
                    f"![결측값 분석 {i+1}]({_rel(plot_path)})",
                    "",
                    "변수별 결측값 현황을 시각화한 차트입니다. 결측값이 많은 변수는 별도의 처리 전략이 필요할 수 있습니다.",
                    ""
                ])
            section_num += 1
        
        # Feature importance plots
        if plot_groups['feature']:
            report_lines.extend([
                f"### 4.{section_num} 특성 중요도 시각화",
                ""
            ])
            for i, plot_path in enumerate(plot_groups['feature']):
                if i > 0:
                    report_lines.append("")
                report_lines.extend([
                    f"![특성 중요도 {i+1}]({_rel(plot_path)})",
                    "",
                    "머신러닝 모델에서 각 특성의 중요도를 시각화한 차트입니다. 상위 특성들이 예측에 가장 큰 영향을 미칩니다.",
                    ""
                ])
            section_num += 1
        
        # Clustering plots
        if plot_groups['clustering']:
            report_lines.extend([
                f"### 4.{section_num} 클러스터링 결과",
                ""
            ])
            for i, plot_path in enumerate(plot_groups['clustering']):
                if i > 0:
                    report_lines.append("")
                report_lines.extend([
                    f"![클러스터링 시각화 {i+1}]({_rel(plot_path)})",
                    "",
                    "클러스터링 알고리즘의 결과를 시각화한 차트입니다. 데이터가 어떻게 그룹화되는지 확인할 수 있습니다.",
                    ""
                ])
            section_num += 1
        
        # Other plots (including any uncategorized)
        if plot_groups['other']:
            report_lines.extend([
                f"### 4.{section_num} 추가 시각화",
                ""
            ])
            for i, plot_path in enumerate(plot_groups['other']):
                if i > 0:
                    report_lines.append("")
                plot_filename = os.path.basename(plot_path)
                report_lines.extend([
                    f"![{plot_filename}]({_rel(plot_path)})",
                    "",
                    "데이터 분석 과정에서 생성된 추가 시각화입니다.",
                    ""
                ])
            section_num += 1
        
        # Summary of all visualizations
        total_plots = sum(len(plots) for plots in plot_groups.values())
        if total_plots > 1:
            report_lines.extend([
                f"총 {total_plots}개의 시각화가 생성되었습니다. 각 차트는 데이터의 다양한 측면을 보여주며, 종합적인 이해를 돕습니다.",
                ""
            ])
    
    # ════════════════════════════════════════════════════════════════════════
    # 5. Generated Code
    # ════════════════════════════════════════════════════════════════════════
    
    if categorized_artifacts['code']:
        report_lines.extend([
            "## 5. 생성된 코드",
            ""
        ])
        
        # Analyze each code file
        code_analyses = []
        for code_path in categorized_artifacts['code']:
            analysis = _extract_code_info(code_path)
            if analysis:
                code_analyses.append(analysis)
        
        # Sort by typical workflow
        def get_priority(analysis):
            filename = analysis.get('filename', '').lower()
            if 'load' in filename:
                return 1
            elif 'eda' in filename:
                return 2
            elif 'automl' in filename:
                return 3
            else:
                return 4
        
        code_analyses.sort(key=get_priority)
        
        # Display each code analysis
        for i, analysis in enumerate(code_analyses, 1):
            report_lines.extend([
                f"### 5.{i} {analysis['description']}",
                "",
                f"**파일명**: `{analysis['filename']}`  ",
                f"**사용된 라이브러리**: `{', '.join(analysis['imports'])}`",
                ""
            ])
            
            if analysis['features']:
                report_lines.extend([
                    "**핵심 기능**:",
                    ""
                ])
                for feature in analysis['features']:
                    report_lines.append(f"- {feature}")
                report_lines.append("")
            
            if analysis['key_lines']:
                report_lines.extend([
                    "**주요 구현 내용**:",
                    ""
                ])
                for line in analysis['key_lines']:
                    report_lines.append(f"- `{line}`")
                report_lines.append("")
            
            if analysis['outputs']:
                report_lines.extend([
                    "**생성되는 산출물**:",
                    ""
                ])
                for output in analysis['outputs']:
                    report_lines.append(f"- {output}")
                report_lines.append("")
    
# ════════════════════════════════════════════════════════════════════════
# 6. Usage Guide
# ════════════════════════════════════════════════════════════════════════

    report_lines.extend([
        "## 6. 산출물 활용 가이드",
        "",
        "> ⚠️ **중요 안내사항**",
        "> ",
        "> 이전까지의 검증된 코드와는 달리, 이 섹션의 모든 코드 예제는 **AI가 자동으로 생성**한 것입니다.",
        "> ",
        "> - ✅ **실행 전 반드시 코드를 검토**하고 환경에 맞게 수정하세요",
        "> - ✅ **파일 경로와 변수명**이 실제 환경과 일치하는지 확인하세요",
        "> - ✅ **필요한 라이브러리가 설치**되어 있는지 확인하세요",
        "> - ✅ **데이터 형식과 구조**가 예제와 일치하는지 검증하세요",
        "> - ✅ **에러 처리 코드를 추가**하여 안정성을 높이세요",
        "> ",
        "> 자동 생성된 코드는 **참고용 템플릿**으로 활용하시고, 실제 프로덕션 환경에서는 충분한 테스트 후 사용하시기 바랍니다.",
        "",
        "---",
        ""
    ])
    
    # Add practical usage examples based on problem type
    if automl_results:
        problem_type = automl_results.get('problem_type', '').lower()
        
        if problem_type == 'classification':
            report_lines.extend([
                "### 6.1 분류 모델 활용",
                "",
                "#### 기본 예측 수행",
                "```python",
                "import pandas as pd",
                "import joblib",
                "",
                "# 모델 로드",
                "model = joblib.load('best_model.pkl')",
                "scaler = joblib.load('scaler.pkl')",
                "label_encoder = joblib.load('label_encoder.pkl')  # 범주형 타겟인 경우",
                "",
                "# 새 데이터 준비",
                "new_data = pd.DataFrame({",
                "    'feature1': [1.5, 2.3],",
                "    'feature2': [0.8, 1.2],",
                "})",
                "",
                "# 전처리 및 예측",
                "new_data_scaled = scaler.transform(new_data)",
                "predictions = model.predict(new_data_scaled)",
                "probabilities = model.predict_proba(new_data_scaled)",
                "",
                "# 라벨 디코딩 (필요한 경우)",
                "if label_encoder:",
                "    predictions = label_encoder.inverse_transform(predictions)",
                "",
                "# 결과 출력",
                "for i, (pred, prob) in enumerate(zip(predictions, probabilities)):",
                "    confidence = prob.max()",
                "    print(f'샘플 {i+1}: 예측={pred}, 신뢰도={confidence:.1%}')",
                "```",
                ""
            ])
            
        elif problem_type == 'regression':
            report_lines.extend([
                "### 6.1 회귀 모델 활용",
                "",
                "#### 예측 및 신뢰구간 계산",
                "```python",
                "import pandas as pd",
                "import joblib",
                "import numpy as np",
                "",
                "# 모델 로드",
                "model = joblib.load('best_model.pkl')",
                "scaler = joblib.load('scaler.pkl')",
                "",
                "# 배치 예측",
                "df = pd.read_csv('new_data.csv')",
                "df_scaled = scaler.transform(df)",
                "predictions = model.predict(df_scaled)",
                "",
                "# 예측 구간 추정 (앙상블 모델의 경우)",
                "if hasattr(model, 'estimators_'):",
                "    tree_preds = [tree.predict(df_scaled) for tree in model.estimators_]",
                "    lower = np.percentile(tree_preds, 5, axis=0)",
                "    upper = np.percentile(tree_preds, 95, axis=0)",
                "    df['prediction'] = predictions",
                "    df['lower_bound'] = lower",
                "    df['upper_bound'] = upper",
                "",
                "# 결과 저장",
                "df.to_csv('predictions.csv', index=False)",
                "```",
                ""
            ])
            
        elif problem_type == 'clustering':
            report_lines.extend([
                "### 6.1 클러스터링 모델 활용",
                "",
                "#### 새로운 데이터 클러스터 할당",
                "```python",
                "import pandas as pd",
                "import joblib",
                "import numpy as np",
                "",
                "# 모델 및 전처리기 로드",
                "model = joblib.load('best_clustering_model.pkl')",
                "scaler = joblib.load('scaler.pkl')",
                "",
                "# 새 데이터 준비",
                "new_data = pd.read_csv('new_data.csv')",
                "numeric_cols = new_data.select_dtypes(include=[np.number]).columns",
                "new_data_scaled = scaler.transform(new_data[numeric_cols])",
                "",
                "# 클러스터 할당",
                "if hasattr(model, 'predict'):",
                "    clusters = model.predict(new_data_scaled)",
                "else:",
                "    # DBSCAN 등의 경우",
                "    clusters = model.fit_predict(new_data_scaled)",
                "",
                "# 결과 추가",
                "new_data['cluster'] = clusters",
                "",
                "# 클러스터별 통계",
                "cluster_stats = new_data.groupby('cluster').mean()",
                "print('클러스터별 평균 특성:')",
                "print(cluster_stats)",
                "",
                "# 시각화",
                "import matplotlib.pyplot as plt",
                "from sklearn.decomposition import PCA",
                "",
                "# PCA로 2차원 시각화",
                "pca = PCA(n_components=2)",
                "X_pca = pca.fit_transform(new_data_scaled)",
                "",
                "plt.figure(figsize=(10, 8))",
                "scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')",
                "plt.colorbar(scatter)",
                "plt.title('클러스터 분포 (PCA 투영)')",
                "plt.xlabel('PC1')",
                "plt.ylabel('PC2')",
                "plt.show()",
                "```",
                ""
            ])
            
        elif problem_type == 'anomaly_detection':
            report_lines.extend([
                "### 6.1 이상치 탐지 모델 활용",
                "",
                "#### 이상치 탐지 및 스코어링",
                "```python",
                "import pandas as pd",
                "import joblib",
                "import numpy as np",
                "",
                "# 모델 로드",
                "model = joblib.load('anomaly_detection_model.pkl')",
                "scaler = joblib.load('scaler.pkl')",
                "",
                "# 새 데이터 준비",
                "new_data = pd.read_csv('new_data.csv')",
                "new_data_scaled = scaler.transform(new_data)",
                "",
                "# 이상치 예측",
                "anomalies = model.predict(new_data_scaled)",
                "# -1: 이상치, 1: 정상",
                "",
                "# 이상치 스코어 (가능한 경우)",
                "if hasattr(model, 'score_samples'):",
                "    anomaly_scores = model.score_samples(new_data_scaled)",
                "    new_data['anomaly_score'] = anomaly_scores",
                "",
                "new_data['is_anomaly'] = anomalies == -1",
                "",
                "# 이상치 비율",
                "anomaly_ratio = (anomalies == -1).sum() / len(anomalies)",
                "print(f'이상치 비율: {anomaly_ratio:.2%}')",
                "",
                "# 이상치 데이터 추출",
                "anomaly_data = new_data[new_data['is_anomaly']]",
                "anomaly_data.to_csv('anomalies.csv', index=False)",
                "```",
                ""
            ])
            
        elif 'timeseries' in problem_type:
            report_lines.extend([
                "### 6.1 시계열 분석 모델 활용",
                "",
                "#### 시계열 예측",
                "```python",
                "import pandas as pd",
                "import joblib",
                "import numpy as np",
                "from datetime import timedelta",
                "",
                "# 모델 로드",
                "model = joblib.load('timeseries_model.pkl')",
                "",
                "# 과거 데이터 준비",
                "historical_data = pd.read_csv('historical_data.csv')",
                "historical_data['date'] = pd.to_datetime(historical_data['date'])",
                "historical_data.set_index('date', inplace=True)",
                "",
                "# 미래 예측 (예: 30일)",
                "forecast_steps = 30",
                "last_date = historical_data.index[-1]",
                "",
                "# 예측 수행",
                "if hasattr(model, 'forecast'):",
                "    forecast = model.forecast(steps=forecast_steps)",
                "else:",
                "    # sklearn 모델의 경우 반복 예측",
                "    forecast = []",
                "    last_values = historical_data.values[-1]",
                "    for i in range(forecast_steps):",
                "        pred = model.predict([last_values])[0]",
                "        forecast.append(pred)",
                "        last_values = np.append(last_values[1:], pred)",
                "",
                "# 예측 결과 데이터프레임",
                "future_dates = pd.date_range(start=last_date + timedelta(days=1), ",
                "                            periods=forecast_steps, freq='D')",
                "forecast_df = pd.DataFrame({",
                "    'date': future_dates,",
                "    'forecast': forecast",
                "})",
                "",
                "# 시각화",
                "import matplotlib.pyplot as plt",
                "",
                "plt.figure(figsize=(12, 6))",
                "plt.plot(historical_data.index, historical_data.values, label='Historical')",
                "plt.plot(forecast_df['date'], forecast_df['forecast'], ",
                "         label='Forecast', linestyle='--')",
                "plt.xlabel('Date')",
                "plt.ylabel('Value')",
                "plt.title('시계열 예측')",
                "plt.legend()",
                "plt.xticks(rotation=45)",
                "plt.tight_layout()",
                "plt.show()",
                "```",
                ""
            ])
            
        elif 'text' in problem_type:
            report_lines.extend([
                "### 6.1 텍스트 분석 모델 활용",
                "",
                "#### 텍스트 분류/분석",
                "```python",
                "import pandas as pd",
                "import joblib",
                "",
                "# 모델 및 전처리기 로드",
                "model = joblib.load('text_model.pkl')",
                "vectorizer = joblib.load('text_vectorizer.pkl')",
                "",
                "# 새 텍스트 데이터",
                "texts = [",
                "    '이 제품은 정말 훌륭합니다!',",
                "    '매우 실망스러운 경험이었습니다.',",
                "    '보통입니다. 특별할 것은 없네요.'",
                "]",
                "",
                "# 텍스트 벡터화",
                "X = vectorizer.transform(texts)",
                "",
                "# 예측",
                "predictions = model.predict(X)",
                "probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None",
                "",
                "# 결과 출력",
                "for i, text in enumerate(texts):",
                "    print(f'텍스트: {text}')",
                "    print(f'예측: {predictions[i]}')",
                "    if probabilities is not None:",
                "        print(f'신뢰도: {probabilities[i].max():.2%}')",
                "    print('-' * 50)",
                "```",
                ""
            ])
            
        elif 'image' in problem_type:
            report_lines.extend([
                "### 6.1 이미지 분석 모델 활용",
                "",
                "#### 이미지 분류/분석",
                "```python",
                "import numpy as np",
                "import joblib",
                "from PIL import Image",
                "import tensorflow as tf",
                "",
                "# 모델 로드",
                "model = tf.keras.models.load_model('image_model.h5')",
                "",
                "# 이미지 전처리 함수",
                "def preprocess_image(image_path, target_size=(224, 224)):",
                "    img = Image.open(image_path)",
                "    img = img.resize(target_size)",
                "    img_array = np.array(img) / 255.0  # 정규화",
                "    return np.expand_dims(img_array, axis=0)",
                "",
                "# 이미지 예측",
                "image_paths = ['image1.jpg', 'image2.jpg']",
                "",
                "for path in image_paths:",
                "    # 전처리",
                "    img_array = preprocess_image(path)",
                "    ",
                "    # 예측",
                "    predictions = model.predict(img_array)",
                "    predicted_class = np.argmax(predictions[0])",
                "    confidence = predictions[0].max()",
                "    ",
                "    print(f'이미지: {path}')",
                "    print(f'예측 클래스: {predicted_class}')",
                "    print(f'신뢰도: {confidence:.2%}')",
                "    print('-' * 50)",
                "```",
                ""
            ])
        else:
            # 기본 가이드
            report_lines.extend([
                "### 6.1 모델 활용 기본 가이드",
                "",
                "생성된 모델 파일을 활용하는 기본적인 방법입니다:",
                "",
                "```python",
                "import pandas as pd",
                "import joblib",
                "",
                "# 모델 로드",
                "model = joblib.load('model.pkl')",
                "",
                "# 데이터 준비",
                "data = pd.read_csv('data.csv')",
                "",
                "# 예측",
                "predictions = model.predict(data)",
                "",
                "# 결과 저장",
                "results = pd.DataFrame({'prediction': predictions})",
                "results.to_csv('results.csv', index=False)",
                "```",
                ""
            ])
    
    # API 서빙 가이드 - 모든 모델 타입에 공통
    if categorized_artifacts['models']:
        report_lines.extend([
            "### 6.2 모델 API 서빙",
            "",
            "FastAPI를 사용한 모델 서빙 예제:",
            "",
            "```python",
            "from fastapi import FastAPI, HTTPException",
            "from pydantic import BaseModel",
            "import joblib",
            "import pandas as pd",
            "import numpy as np",
            "",
            "app = FastAPI()",
            "",
            "# 모델 로드 (서버 시작시 한번만)",
            "model = joblib.load('best_model.pkl')",
            "scaler = joblib.load('scaler.pkl')",
            "",
            "class PredictionRequest(BaseModel):",
            "    features: dict",
            "",
            "class PredictionResponse(BaseModel):",
            "    prediction: float",
            "    confidence: float = None",
            "",
            "@app.post('/predict', response_model=PredictionResponse)",
            "async def predict(request: PredictionRequest):",
            "    try:",
            "        # 데이터프레임 변환",
            "        df = pd.DataFrame([request.features])",
            "        ",
            "        # 전처리",
            "        df_scaled = scaler.transform(df)",
            "        ",
            "        # 예측",
            "        prediction = model.predict(df_scaled)[0]",
            "        ",
            "        # 신뢰도 (가능한 경우)",
            "        confidence = None",
            "        if hasattr(model, 'predict_proba'):",
            "            proba = model.predict_proba(df_scaled)[0]",
            "            confidence = float(proba.max())",
            "        ",
            "        return PredictionResponse(",
            "            prediction=float(prediction),",
            "            confidence=confidence",
            "        )",
            "    except Exception as e:",
            "        raise HTTPException(status_code=400, detail=str(e))",
            "",
            "# 실행: uvicorn main:app --reload",
            "```",
            ""
        ])
    
    # 전체 분석 재현 가이드
    report_lines.extend([
        "### 6.3 전체 분석 재현",
        "",
        "생성된 코드를 순차적으로 실행하여 전체 분석을 재현할 수 있습니다:",
        "",
        "```bash",
        "# 1. 필요 패키지 설치",
        "pip install pandas numpy matplotlib seaborn scikit-learn joblib",
        ""
    ])
    
    # 문제 타입별 추가 패키지
    if automl_results:
        problem_type = automl_results.get('problem_type', '').lower()
        if 'timeseries' in problem_type:
            report_lines.append("pip install statsmodels  # 시계열 분석용")
        elif 'text' in problem_type:
            report_lines.append("pip install nltk scikit-learn  # 텍스트 분석용")
        elif 'image' in problem_type:
            report_lines.append("pip install tensorflow pillow  # 이미지 분석용")
        
        # XGBoost 등 추가 패키지
        if XGB_AVAILABLE:
            report_lines.append("pip install xgboost  # 고급 부스팅 모델")
        if LGB_AVAILABLE:
            report_lines.append("pip install lightgbm  # 고급 부스팅 모델")
    
    report_lines.extend([
        "",
        "# 2. 코드 실행 (순서대로)",
        "python load_*.py      # 데이터 로딩",
        "python eda_*.py       # 탐색적 데이터 분석",
        "python automl_*.py    # 머신러닝 모델링",
        "",
        "# 3. 결과 확인",
        "ls *.png  # 생성된 시각화",
        "ls *.pkl  # 저장된 모델",
        "ls *.csv  # 처리된 데이터",
        "```",
        ""
    ])
    
    # 모니터링 가이드
    report_lines.extend([
        "### 6.4 모델 모니터링",
        "",
        "프로덕션 환경에서 모델 성능을 모니터링하는 방법:",
        "",
        "```python",
        "import pandas as pd",
        "import numpy as np",
        "from datetime import datetime",
        "import joblib",
        "",
        "class ModelMonitor:",
        "    def __init__(self, model_path, threshold=0.1):",
        "        self.model = joblib.load(model_path)",
        "        self.threshold = threshold",
        "        self.predictions_log = []",
        "        ",
        "    def predict_and_log(self, X):",
        "        # 예측",
        "        prediction = self.model.predict(X)",
        "        ",
        "        # 로깅",
        "        self.predictions_log.append({",
        "            'timestamp': datetime.now(),",
        "            'input_shape': X.shape,",
        "            'prediction': prediction",
        "        })",
        "        ",
        "        return prediction",
        "    ",
        "    def check_drift(self, recent_performance):",
        "        # 성능 저하 감지",
        "        if recent_performance < self.baseline_performance - self.threshold:",
        "            print('⚠️ 성능 저하 감지! 모델 재학습이 필요할 수 있습니다.')",
        "            return True",
        "        return False",
        "```",
        ""
    ])
    
    # ════════════════════════════════════════════════════════════════════════
    # 7. Conclusions and Recommendations
    # ════════════════════════════════════════════════════════════════════════
    
    report_lines.extend([
        "## 7. 결론 및 권장사항",
        ""
    ])
    
    # Generate conclusions
    conclusions = []
    recommendations = []
    
    # Model performance conclusions
    if automl_results and automl_results.get('best_model'):
        best_model = automl_results['best_model']
        model_name = best_model.get('name', 'N/A')
        scores = best_model.get('scores', {})
        
        if automl_results.get('problem_type') == 'classification':
            acc = scores.get('test_accuracy', 0)
            if acc > 0.95:
                conclusions.append(f"**{model_name}** 모델이 {_format_percentage(acc)}의 매우 높은 정확도를 달성했습니다.")
                recommendations.append("과적합 가능성을 확인하기 위해 추가 검증 데이터셋에서 평가를 권장합니다.")
            elif acc > 0.85:
                conclusions.append(f"**{model_name}** 모델이 {_format_percentage(acc)}의 우수한 성능을 보였습니다.")
                recommendations.append("하이퍼파라미터 튜닝을 통해 추가적인 성능 향상이 가능할 수 있습니다.")
            else:
                conclusions.append(f"모델 성능이 {_format_percentage(acc)}로 개선이 필요합니다.")
                recommendations.append("특성 엔지니어링과 데이터 품질 개선을 검토하세요.")
    
    # Data quality conclusions
    if eda_results:
        analysis = eda_results.get('analysis', {})
        basic_info = analysis.get('basic_info', {})
        
        if basic_info.get('missing_values'):
            total_missing = sum(basic_info['missing_values'].values())
            total_cells = basic_info.get('shape', [1, 1])[0] * basic_info.get('shape', [1, 1])[1]
            missing_ratio = (total_missing / total_cells * 100) if total_cells > 0 else 0
            
            if missing_ratio > 20:
                conclusions.append(f"데이터 품질 이슈가 확인되었습니다 (결측률 {missing_ratio:.1f}%).")
                recommendations.append("고급 결측값 대체 기법(MICE, KNN 등)을 고려해보세요.")
    
    # Feature importance conclusions
    if automl_results and automl_results.get('feature_importance'):
        conclusions.append("특성 중요도 분석을 통해 핵심 예측 변수를 식별했습니다.")
        recommendations.append("중요도가 낮은 특성을 제거하여 모델을 단순화할 수 있습니다.")
    
    # Default recommendations
    recommendations.extend([
        "정기적인 모델 재학습을 통해 데이터 드리프트에 대응하세요.",
        "모델 배포 전 A/B 테스트를 통한 실제 비즈니스 임팩트를 검증하세요.",
        "모델 모니터링 시스템을 구축하여 성능 저하를 조기에 감지하세요."
    ])
    
    # Write conclusions
    report_lines.extend([
        "### 주요 결론",
        ""
    ])
    
    for conclusion in conclusions:
        report_lines.append(f"- {conclusion}")
    
    if not conclusions:
        report_lines.append("- 분석이 성공적으로 완료되었으며, 데이터의 특성과 패턴을 파악했습니다.")
    
    report_lines.extend([
        "",
        "### 권장사항",
        ""
    ])
    
    for rec in recommendations:
        report_lines.append(f"- {rec}")
    
    report_lines.extend([
        "",
        "---",
        "",
        f"**보고서 생성 완료**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**분석 도구**: Advanced Data Science MCP Server  ",
        f"**보고서 버전**: Enhanced v3.0  ",
        f"**보고서 ID**: `{report_id}`",
        ""
    ])
    
    # Join all lines
    final_report = "\n".join(report_lines)
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"comprehensive_report_{report_id}_{timestamp}.md"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(final_report)
    
    # Log operation
    operation_id = report_id  # Use report_id as operation_id
    
    tracker.log_operation(
        operation_id=operation_id,
        operation_type="create_comprehensive_report",
        input_data={
            "operations": operations,
            "report_title": report_title,
            "dataset_name": dataset_name
        },
        output_data={
            "report_id": report_id,
            "report_path": report_path,
            "report_filename": report_filename,
            "report_size_kb": len(final_report) / 1024
        },
        files_created=[report_path]
    )
    
    return json.dumps({
        "status": "success",
        "report_id": report_id,
        "operation_id": operation_id,
        "report_path": report_path,
        "report_filename": report_filename,
        "report_size_kb": round(len(final_report) / 1024, 2),
        "message": f"보고서가 성공적으로 생성되었습니다: {report_filename}",
        "location": REPORTS_DIR
    }, indent=2, ensure_ascii=False)

# Initialize FastMCP server
mcp = FastMCP(
    "AdvancedDataScience",
    instructions="""
An advanced data science server providing comprehensive ML/DL capabilities:

🚀 **Core Features:**
- Smart data loading with adaptive sampling strategies
- Advanced AutoML with problem type detection
- Deep learning with AutoKeras integration  
- High-performance algorithms (XGBoost, LightGBM, CatBoost)
- Gaussian Processes, Spectral Clustering, HDBSCAN
- Advanced dimensionality reduction (t-SNE, UMAP, Isomap)
- Model interpretation with SHAP values
- Interactive visualizations with Plotly
- Complete code generation for reproducibility
- Intelligent feature engineering

🧠 **Problem Type Detection:**
- Classification (binary/multiclass)
- Regression (linear/non-linear)
- Clustering (density/partition-based)
- Time series analysis
- Anomaly detection
- Text analysis
- Image analysis

🔬 **Advanced Analytics:**
- Silhouette analysis for optimal clusters
- Outlier detection and treatment
- Feature importance analysis
- Cross-validation with multiple metrics
- Ensemble methods
- Hyperparameter optimization
- Imbalanced data handling

📊 **Visualization Suite:**
- Advanced dashboards
- Dimensionality reduction plots
- Interactive Plotly charts
- Feature importance visualizations
- Model performance comparisons

All operations are tracked with comprehensive logging and generate publication-ready reports.
    """
)

@mcp.tool()
async def health_check() -> str:
    """
    Health check for the Advanced Data Science MCP server.
    
    Returns:
        str: Server health status and environment information
    """
    try:
        # Ensure sandbox structure exists
        if not ensure_sandbox_structure():
            return json.dumps({"status": "error", "message": "Failed to ensure sandbox directory structure"})
        
        import platform
        
        health_info = {
            "status": "healthy",
            "message": "Advanced Data Science MCP Server is running",
            "server_info": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "sandbox_directory": SANDBOX_DIR,
                "server_port": SERVER_PORT
            },
            "core_packages": {
                "pandas": True,
                "numpy": True,
                "matplotlib": True,
                "seaborn": True,
                "scikit_learn": SKLEARN_AVAILABLE,
                "plotly": PLOTLY_AVAILABLE
            },
            "ml_packages": {
                "tensorflow": TF_AVAILABLE,
                "xgboost": XGB_AVAILABLE,
                "lightgbm": LGB_AVAILABLE,
                "catboost": CATBOOST_AVAILABLE,
                "autokeras": AUTOKERAS_AVAILABLE,
                "keras_tuner": KERAS_TUNER_AVAILABLE
            },
            "advanced_packages": {
                "statsmodels": STATSMODELS_AVAILABLE,
                "shap": SHAP_AVAILABLE,
                "umap": UMAP_AVAILABLE,
                "hdbscan": HDBSCAN_AVAILABLE,
                "imbalanced_learn": IMBLEARN_AVAILABLE,
                "optuna": OPTUNA_AVAILABLE
            },
            "features": [
                "Smart data loading with adaptive sampling",
                "Advanced AutoML with problem detection",
                "Deep learning with AutoKeras",
                "Gaussian Processes and advanced algorithms",
                "Clustering with silhouette optimization",
                "Advanced dimensionality reduction",
                "Model interpretation with SHAP",
                "Interactive Plotly visualizations",
                "Complete Python code generation",
                "Comprehensive markdown reports"
            ],
            "dataset_count": len(list(Path(DATASETS_DIR).glob("*.csv"))) if os.path.exists(DATASETS_DIR) else 0,
            "operations_logged": len(tracker.operations),
            "code_generation": "Enabled",
            "visualization_formats": ["PNG", "HTML", "Interactive"]
        }
        
        logger.info("Health check completed successfully")
        return json.dumps(health_info, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Health check failed: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"status": "error", "message": error_msg})

@mcp.tool()
async def load_dataset(file_path: str, target_column: str = None, operation_id: str = None) -> str:
    """
    Load a dataset from file and register it for tracking.
    Automatically applies smart sampling for large datasets (>100MB).
    Auto-detects problem type and suggests target columns.
    
    Args:
        file_path (str): Path to the dataset file (CSV, Excel, etc.)
        target_column (str): Target column name for stratified sampling (optional)
        operation_id (str): Optional operation ID for tracking
        
    Returns:
        str: Dataset summary and registration info with sampling details
    """
    if operation_id is None:
        operation_id = f"load_{uuid.uuid4().hex[:8]}"
    
    try:
        # Ensure sandbox structure exists
        if not ensure_sandbox_structure():
            error_msg = "Failed to ensure sandbox directory structure"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        logger.info(f"Loading dataset: {file_path}")
        
        # Check if file exists and get size
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        file_size_mb = get_file_size_mb(file_path)
        logger.info(f"File size: {file_size_mb:.1f}MB")
        
        # Determine file type and load
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            df_original = pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            df_original = pd.read_excel(file_path)
        elif file_ext == '.json':
            df_original = pd.read_json(file_path)
        elif file_ext == '.parquet':
            df_original = pd.read_parquet(file_path)
        else:
            error_msg = f"Unsupported file format: {file_ext}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        # Validate loaded data
        if df_original.empty:
            error_msg = "Loaded dataset is empty"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        # Auto-detect problem type
        detected_problem_type = auto_detect_problem_type(df_original, target_column)
        logger.info(f"Detected problem type: {detected_problem_type}")
        
        # Analyze supervised learning potential
        supervised_analysis = analyze_supervised_learning_potential(df_original)
        
        # 샘플링 전략 결정
        sampling_strategy = determine_sampling_strategy(df_original, file_size_mb, target_column)
        logger.info(f"Sampling strategy: {sampling_strategy}")
        
        # 샘플링 적용
        df_final, sampling_details = apply_sampling(df_original, sampling_strategy, target_column)
        
        # Register dataset (최종 처리된 데이터로)
        dataset_id = f"dataset_{operation_id}"
        tracker.register_dataset(dataset_id, df_final, file_path, sampling_details)
        
        # Save processed dataset to sandbox for future use
        dataset_file = os.path.join(DATASETS_DIR, f"{dataset_id}.csv")
        df_final.to_csv(dataset_file, index=False)
        
        # Generate loading code with sampling logic
        basic_loading_code = f'''
import pandas as pd
import numpy as np

# Original file info: {file_size_mb:.1f}MB, {len(df_original)} rows
'''
        
        # Add sampling code if applied
        if sampling_details.get("applied", False):
            sampling_code = generate_sampling_code(dataset_id, sampling_details, target_column)
            loading_code = basic_loading_code + sampling_code
        else:
            loading_code = basic_loading_code + f'''
# Load full dataset (no sampling needed)
df = pd.read_{file_ext[1:]}('{file_path}')
'''
        
        loading_code += '''
print("Dataset loading completed!")
print(f"Final dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
'''
        
        # Save generated code
        code_path = save_code_as_file(operation_id, loading_code, "data_loading")
        
        # Generate safe summary
        summary = safe_dataframe_summary(df_final)
        
        # Prepare comprehensive result
        result = {
            "operation_id": operation_id,
            "dataset_id": dataset_id,
            "status": "success",
            "file_info": {
                "original_file": file_path,
                "file_size_mb": round(file_size_mb, 2),
                "original_rows": len(df_original),
                "original_columns": len(df_original.columns)
            },
            "problem_analysis": {
                "detected_type": detected_problem_type,
                "supervised_potential": supervised_analysis["has_potential"],
                "recommended_targets": supervised_analysis["recommended_targets"][:3] if supervised_analysis["has_potential"] else []
            },
            "sampling_info": {
                "strategy": sampling_strategy,
                "details": sampling_details,
                "final_rows": len(df_final),
                "data_reduction": f"{(1 - len(df_final)/len(df_original))*100:.1f}%" if sampling_details.get("applied") else "0%"
            },
            "summary": summary,
            "dataset_file": dataset_file,
            "generated_code": code_path if code_path else None,
            "recommendations": []
        }
        
        # Add recommendations based on analysis
        recommendations = []
        
        # Problem type recommendations
        if detected_problem_type == 'classification':
            recommendations.append("🎯 Classification problem detected - use auto_ml_pipeline for best models")
        elif detected_problem_type == 'regression':
            recommendations.append("📈 Regression problem detected - consider non-linear models")
        elif detected_problem_type == 'clustering':
            recommendations.append("🔗 Clustering analysis suggested - will auto-optimize cluster count")
        elif detected_problem_type.startswith('timeseries'):
            recommendations.append("📅 Time series data detected - specialized models available")
        elif detected_problem_type.endswith('_analysis'):
            recommendations.append("🔍 Specialized analysis type detected - custom processing available")
        
        # Sampling recommendations
        if sampling_details.get("applied", False):
            recommendations.append(f"✅ Applied {sampling_details['method']} sampling to reduce dataset size for better performance")
            recommendations.append(f"📊 Sample quality: {sampling_details.get('quality_check', {}).get('overall_quality', 'good')}")
            recommendations.append("🔍 All analysis results are based on the sampled data")
            recommendations.append("⚠️  For production use, consider validating results with the full dataset")
            recommendations.append("💡 The generated code includes the same sampling logic for reproducibility")
            
            if sampling_details.get('quality_check', {}).get('overall_quality') == 'excellent':
                recommendations.append("🎯 Sample represents original data very well - results should be highly reliable")
            elif sampling_details.get('quality_check', {}).get('overall_quality') == 'fair':
                recommendations.append("⚠️  Sample quality is fair - interpret results with caution")
        else:
            recommendations.append("✅ Using full dataset - no sampling applied")
        
        # Target column recommendations
        if supervised_analysis["has_potential"] and not target_column:
            best_target = supervised_analysis["recommended_targets"][0]
            recommendations.append(f"🎯 Suggested target column: '{best_target['column']}' ({best_target['suggested_type']})")
        
        result["recommendations"] = recommendations
        
        # Log operation
        files_created = [dataset_file]
        if code_path:
            files_created.append(code_path)
            
        tracker.log_operation(
            operation_id=operation_id,
            operation_type="load_dataset",
            input_data={
                "file_path": file_path,
                "target_column": target_column,
                "file_size_mb": file_size_mb
            },
            output_data=result,
            files_created=files_created
        )
        
        logger.info(f"Dataset loaded successfully: {dataset_id} (sampled: {sampling_details.get('applied', False)})")
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except pd.errors.EmptyDataError:
        error_msg = f"Empty data file: {file_path}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg, "operation_id": operation_id})
    except pd.errors.ParserError as e:
        error_msg = f"Error parsing file {file_path}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg, "operation_id": operation_id})
    except MemoryError:
        error_msg = f"File too large to load into memory: {file_path}. Try with a smaller file or increase system memory."
        logger.error(error_msg)
        return json.dumps({"error": error_msg, "operation_id": operation_id})
    except Exception as e:
        error_msg = f"Error loading dataset {file_path}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return json.dumps({"error": error_msg, "operation_id": operation_id})

@mcp.tool()
async def perform_eda(dataset_id: str, operation_id: str = None) -> str:
    """
    Perform Advanced Exploratory Data Analysis (EDA) on a dataset.
    
    Args:
        dataset_id (str): ID of the registered dataset
        operation_id (str): Optional operation ID for tracking
        
    Returns:
        str: EDA results with generated plots and insights
    """
    if operation_id is None:
        operation_id = f"eda_{uuid.uuid4().hex[:8]}"
    
    try:
        # Ensure sandbox structure exists
        if not ensure_sandbox_structure():
            error_msg = "Failed to ensure sandbox directory structure"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        logger.info(f"Performing advanced EDA on dataset: {dataset_id}")
        
        # Load dataset
        dataset_file = os.path.join(DATASETS_DIR, f"{dataset_id}.csv")
        if not os.path.exists(dataset_file):
            error_msg = f"Dataset {dataset_id} not found"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        try:
            df = pd.read_csv(dataset_file)
        except Exception as e:
            error_msg = f"Error reading dataset file: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        if df.empty:
            error_msg = "Dataset is empty"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        # Get sampling information for this dataset
        sampling_info = get_dataset_sampling_info(dataset_id)
        logger.info(f"Retrieved sampling info for {dataset_id}: {sampling_info.get('applied', False)}")
        
        # Create advanced visualizations
        plot_paths = create_advanced_visualizations(df, operation_id)
        
        # Generate EDA code with sampling information
        eda_code = generate_eda_code(dataset_id, sampling_info)
        code_path = save_code_as_file(operation_id, eda_code, "eda")
        
        # Prepare EDA results
        eda_results = {
            "operation_id": operation_id,
            "dataset_id": dataset_id,
            "analysis": {},
            "plots_generated": plot_paths,
            "insights": [],
            "recommendations": [],
            "generated_code": code_path if code_path else None,
            "status": "success"
        }
        
        # Add sampling information to results if available
        if sampling_info and sampling_info.get("applied", False):
            eda_results["sampling_info"] = {
                "data_is_sampled": True,
                "original_rows": sampling_info.get("original_rows"),
                "current_rows": len(df),
                "sampling_method": sampling_info.get("method"),
                "sample_quality": sampling_info.get("quality_check", {}).get("overall_quality"),
                "note": "All analysis results are based on sampled data"
            }
        else:
            eda_results["sampling_info"] = {
                "data_is_sampled": False,
                "note": "Analysis performed on full dataset"
            }
        
        # Basic statistics with safe JSON serialization
        try:
            eda_results["analysis"]["basic_info"] = safe_dataframe_summary(df, sampling_info=sampling_info)
        except Exception as e:
            logger.warning(f"Error generating basic info: {e}")
            eda_results["analysis"]["basic_info"] = {"error": "Failed to generate basic info"}
        
        # Correlation analysis for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            try:
                correlation_matrix = df[numeric_cols].corr()
                # Convert to JSON-serializable format safely
                eda_results["analysis"]["correlation"] = safe_json_serialize(correlation_matrix)
                
                # Find strong correlations
                strong_correlations = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i + 1, len(correlation_matrix.columns)):
                        corr_val = correlation_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7 and not pd.isna(corr_val):
                            strong_correlations.append({
                                'var1': correlation_matrix.columns[i],
                                'var2': correlation_matrix.columns[j],
                                'correlation': float(corr_val),
                                'strength': 'very strong' if abs(corr_val) > 0.9 else 'strong'
                            })
                
                eda_results["analysis"]["strong_correlations"] = strong_correlations
                
            except Exception as e:
                logger.warning(f"Error in correlation analysis: {e}")
                eda_results["analysis"]["correlation_error"] = str(e)
        
        # Outlier analysis
        outlier_analysis = {}
        for col in numeric_cols:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_analysis[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
            except Exception as e:
                logger.warning(f"Error in outlier analysis for {col}: {e}")
        
        eda_results["analysis"]["outlier_analysis"] = outlier_analysis
        
        # Data type analysis
        data_types_analysis = {
            'numeric': {
                'count': len(numeric_cols),
                'columns': list(numeric_cols),
                'statistics': {}
            },
            'categorical': {
                'count': len(df.select_dtypes(include=['object', 'category']).columns),
                'columns': list(df.select_dtypes(include=['object', 'category']).columns),
                'cardinality': {}
            },
            'datetime': {
                'count': len(df.select_dtypes(include=['datetime']).columns),
                'columns': list(df.select_dtypes(include=['datetime']).columns)
            }
        }
        
        # Analyze categorical variables
        for col in data_types_analysis['categorical']['columns']:
            unique_count = df[col].nunique()
            data_types_analysis['categorical']['cardinality'][col] = {
                'unique_values': unique_count,
                'cardinality_level': 'high' if unique_count > 50 else 'medium' if unique_count > 10 else 'low'
            }
        
        eda_results["analysis"]["data_types"] = data_types_analysis
        
        # Generate insights
        insights = []
        
        try:
            # Add sampling-related insights first
            if sampling_info and sampling_info.get("applied", False):
                insights.append(f"📊 Data Analysis: Based on {sampling_info.get('method', 'unknown')} sampled data ({len(df):,} from {sampling_info.get('original_rows', 'unknown'):,} rows)")
                quality = sampling_info.get('quality_check', {}).get('overall_quality', 'unknown')
                if quality == 'excellent':
                    insights.append("✅ Sample quality: Excellent - results highly representative of full dataset")
                elif quality == 'good':
                    insights.append("✅ Sample quality: Good - results should be reliable")
                elif quality == 'fair':
                    insights.append("⚠️ Sample quality: Fair - interpret results with caution")
                insights.append("💡 For production decisions, validate findings with full dataset")
            else:
                insights.append("✅ Analysis performed on complete dataset")
            
            # Data quality insights
            missing_ratio = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            if missing_ratio > 20:
                insights.append(f"⚠️ High missing data ratio: {missing_ratio:.1f}% - advanced imputation needed")
            elif missing_ratio > 5:
                insights.append(f"📉 Moderate missing data: {missing_ratio:.1f}% - consider imputation strategies")
            elif missing_ratio > 0:
                insights.append(f"📉 Low missing data: {missing_ratio:.1f}% - minimal preprocessing needed")
            else:
                insights.append("✅ No missing values detected - excellent data quality")
            
            # Correlation insights
            if "strong_correlations" in eda_results["analysis"] and eda_results["analysis"]["strong_correlations"]:
                strong_corr_count = len(eda_results["analysis"]["strong_correlations"])
                insights.append(f"🔗 Found {strong_corr_count} strong correlations - feature selection opportunity")
            
            # Outlier insights
            high_outlier_cols = [col for col, info in outlier_analysis.items() if info['percentage'] > 5]
            if high_outlier_cols:
                insights.append(f"📊 High outlier columns: {', '.join(high_outlier_cols[:3])} - consider robust methods")
            
            # Dimensionality insights
            if len(numeric_cols) > 20:
                insights.append("📈 High-dimensional data - dimensionality reduction recommended")
            elif len(numeric_cols) > 10:
                insights.append("📊 Moderate dimensions - feature selection may improve model performance")
            
            # Data size insights
            current_size = len(df)
            if current_size < 100:
                insights.append("📊 Small dataset - consider simple models to avoid overfitting")
            elif current_size > 100000:
                insights.append("📊 Large dataset - suitable for deep learning and complex models")
            
            # Cardinality insights
            high_cardinality_cols = [col for col, info in data_types_analysis['categorical']['cardinality'].items() 
                                   if info['cardinality_level'] == 'high']
            if high_cardinality_cols:
                insights.append(f"🏷️ High cardinality columns: {', '.join(high_cardinality_cols[:3])} - need advanced encoding")
            
        except Exception as e:
            logger.warning(f"Error generating insights: {e}")
            insights.append("⚠️ Unable to generate some automatic insights due to data processing error")
        
        eda_results["insights"] = insights
        
        # Generate recommendations
        recommendations = []
        
        # Analysis recommendations
        if len(numeric_cols) > 2:
            recommendations.append("🔬 Consider advanced dimensionality reduction (t-SNE, UMAP)")
        
        if len(df) > 1000 and len(numeric_cols) > 5:
            recommendations.append("🎯 Dataset suitable for advanced machine learning algorithms")
        
        # Model recommendations based on data characteristics
        categorical_ratio = len(data_types_analysis['categorical']['columns']) / len(df.columns)
        if categorical_ratio > 0.5:
            recommendations.append("🏷️ High categorical ratio - consider CatBoost or target encoding")
        
        if any(info['cardinality_level'] == 'high' for info in data_types_analysis['categorical']['cardinality'].values()):
            recommendations.append("🔧 High cardinality detected - use frequency or target encoding")
        
        # Visualization recommendations
        if len(numeric_cols) >= 2:
            recommendations.append("📊 Create pair plots for deeper variable relationships")
        
        if len(plot_paths) > 0:
            recommendations.append(f"🎨 {len(plot_paths)} advanced visualizations generated - review for patterns")
        
        # Preprocessing recommendations
        if missing_ratio > 0:
            recommendations.append("🔧 Apply advanced imputation strategies for missing values")
        
        if high_outlier_cols:
            recommendations.append("📊 Consider outlier treatment or robust algorithms")
        
        eda_results["recommendations"] = recommendations
        
        # Save EDA report
        try:
            report_file = os.path.join(REPORTS_DIR, f"eda_report_{operation_id}.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(eda_results, f, indent=2, ensure_ascii=False)
            
            # Log operation
            files_created = [report_file] + plot_paths
            if code_path:
                files_created.append(code_path)
                
            tracker.log_operation(
                operation_id=operation_id,
                operation_type="perform_eda",
                input_data={"dataset_id": dataset_id},
                output_data=eda_results,
                files_created=files_created
            )
        except Exception as e:
            logger.warning(f"Error saving EDA report: {e}")
        
        logger.info(f"Advanced EDA completed successfully: {operation_id}")
        return json.dumps(eda_results, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error performing EDA on {dataset_id}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        error_result = {
            "operation_id": operation_id,
            "dataset_id": dataset_id,
            "status": "error",
            "error": error_msg
        }
        return json.dumps(error_result, indent=2, ensure_ascii=False)

@mcp.tool()
async def auto_ml_pipeline(dataset_id: str, target_column: str = None, 
                          test_size: float = 0.2, cv_folds: int = 5,
                          include_advanced: bool = True,
                          operation_id: str = None) -> str:
    """
    Run advanced automated machine learning pipeline end-to-end.
    Automatically detects supervised learning potential and suggests target columns.
    Includes advanced algorithms and optimization techniques.
    
    Args:
        dataset_id (str): ID of the registered dataset
        target_column (str): Target column name (None, "null", "none" for auto-detection or unsupervised)
        test_size (float): Test set size ratio (default: 0.2)
        cv_folds (int): Cross-validation folds (default: 5)
        include_advanced (bool): Include advanced models like XGBoost, LightGBM, CatBoost
        operation_id (str): Optional operation ID for tracking
        
    Returns:
        str: AutoML results with model performance and recommendations
    """
    if operation_id is None:
        operation_id = f"automl_{uuid.uuid4().hex[:8]}"
    
    try:
        # Ensure sandbox structure exists
        if not ensure_sandbox_structure():
            error_msg = "Failed to ensure sandbox directory structure"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        logger.info(f"Running advanced AutoML pipeline on dataset: {dataset_id}")
        
        # Load dataset
        dataset_file = os.path.join(DATASETS_DIR, f"{dataset_id}.csv")
        if not os.path.exists(dataset_file):
            error_msg = f"Dataset {dataset_id} not found"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        df = pd.read_csv(dataset_file)
        
        if not SKLEARN_AVAILABLE:
            error_msg = "scikit-learn not available for AutoML"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        # Handle null/none string inputs
        if target_column and target_column.lower() in ['null', 'none', 'nan', '']:
            target_column = None
        
        # Auto-detect problem type
        problem_type = auto_detect_problem_type(df, target_column)
        logger.info(f"Detected problem type: {problem_type}")
        
        # Analyze supervised learning potential
        supervised_analysis = analyze_supervised_learning_potential(df)
        
        # Determine problem type and target column
        if target_column is None:
            if supervised_analysis["has_potential"] and problem_type not in ['timeseries', 'image_analysis', 'text_analysis']:
                # Suggest the best target column
                best_candidate = supervised_analysis["analysis_details"]["best_candidate"]
                
                # Create suggestion message
                suggestion_msg = {
                    "message": "🎯 Supervised Learning Potential Detected!",
                    "supervised_analysis": supervised_analysis,
                    "recommendation": f"Consider using '{best_candidate['column']}' as target for {best_candidate['suggested_type']}",
                    "auto_selected": False,
                    "proceeding_with": "unsupervised clustering",
                    "note": "To use supervised learning, specify target_column parameter"
                }
                
                # Still proceed with clustering but inform user of supervised potential
                problem_type = 'clustering'
                target_column = None
                logger.info(f"Supervised potential found but proceeding with clustering: {best_candidate['column']}")
            else:
                if problem_type not in ['clustering', 'anomaly_detection', 'timeseries', 'image_analysis', 'text_analysis']:
                    problem_type = 'clustering'
                target_column = None
                logger.info(f"No suitable target columns found, proceeding with {problem_type}")
        else:
            # Validate specified target column
            if target_column not in df.columns:
                error_msg = f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}"
                logger.error(error_msg)
                return json.dumps({"error": error_msg, "operation_id": operation_id})
            
            # Determine problem type based on target column
            target = df[target_column]
            if target.dtype.kind in 'biufc':  # numeric
                unique_ratio = len(target.unique()) / len(target)
                if unique_ratio > 0.05 or len(target.unique()) > 20:
                    problem_type = 'regression'
                else:
                    problem_type = 'classification'
            else:
                problem_type = 'classification'
        
        # Get sampling information for this dataset
        sampling_info = get_dataset_sampling_info(dataset_id)
        logger.info(f"Retrieved sampling info for AutoML {dataset_id}: {sampling_info.get('applied', False)}")
        
        # Generate AutoML code with sampling information
        automl_code = generate_automl_code(dataset_id, target_column or 'target', problem_type, sampling_info)
        code_path = save_code_as_file(operation_id, automl_code, "automl")
        
        # AutoML Results
        automl_results = {
            "operation_id": operation_id,
            "dataset_id": dataset_id,
            "target_column": target_column,
            "problem_type": problem_type,
            "supervised_analysis": supervised_analysis,
            "preprocessing_steps": [],
            "model_results": {},
            "best_model": None,
            "feature_importance": None,
            "plots_generated": [],
            "recommendations": [],
            "generated_code": code_path if code_path else None,
            "performance_summary": {}
        }
        
        # Add suggestion message if supervised potential was found but clustering was chosen
        if target_column is None and supervised_analysis["has_potential"]:
            automl_results["supervised_suggestion"] = suggestion_msg
        
        # Add sampling information to results if available
        if sampling_info and sampling_info.get("applied", False):
            automl_results["sampling_info"] = {
                "data_is_sampled": True,
                "original_rows": sampling_info.get("original_rows"),
                "training_rows": len(df),
                "sampling_method": sampling_info.get("method"),
                "sample_quality": sampling_info.get("quality_check", {}).get("overall_quality"),
                "note": "Model trained on sampled data for performance optimization"
            }
        else:
            automl_results["sampling_info"] = {
                "data_is_sampled": False,
                "note": "Model trained on full dataset"
            }
        
        # Data preprocessing
        processed_df = df.copy()
        preprocessing_steps = []
        
        # Handle missing values
        missing_cols = df.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        if len(missing_cols) > 0:
            for col in missing_cols.index:
                if df[col].dtype.kind in 'biufc':  # numeric
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                    preprocessing_steps.append(f"{col}: filled with median")
                else:  # categorical
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0] if not processed_df[col].mode().empty else 'missing')
                    preprocessing_steps.append(f"{col}: filled with mode")
        
        # Handle categorical variables
        categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
        if target_column and target_column in categorical_cols:
            categorical_cols = categorical_cols.drop(target_column)
        
        for col in categorical_cols:
            if processed_df[col].nunique() > 10:
                # Frequency encoding for high cardinality
                freq_encoding = processed_df[col].value_counts().to_dict()
                processed_df[col] = processed_df[col].map(freq_encoding)
                preprocessing_steps.append(f"{col}: frequency encoding")
            else:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(processed_df[col], prefix=col)
                processed_df = pd.concat([processed_df.drop(col, axis=1), dummies], axis=1)
                preprocessing_steps.append(f"{col}: one-hot encoding")
        
        automl_results["preprocessing_steps"] = preprocessing_steps
        
        if problem_type in ['classification', 'regression'] and target_column:
            # Supervised learning
            X = processed_df.drop(target_column, axis=1)
            y = processed_df[target_column]
            
            # Handle target encoding for classification
            label_encoder = None
            if problem_type == 'classification' and y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            
            # Feature scaling
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
            
            # Train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42,
                stratify=y if problem_type == 'classification' else None
            )
            
            # Create models
            models = create_advanced_models(problem_type, X.shape[1], 
                                          len(np.unique(y)) if problem_type == 'classification' else None,
                                          len(df))
            
            # Remove advanced models if not requested
            if not include_advanced:
                advanced_model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'Gaussian Process', 'QDA']
                for name in advanced_model_names:
                    models.pop(name, None)
            
            # Model training and evaluation
            model_scores = {}
            best_score = -np.inf if problem_type == 'regression' else 0
            best_model_info = None
            
            for name, model in models.items():
                try:
                    logger.info(f"Training {name}...")
                    
                    # Cross-validation
                    scoring = 'neg_mean_squared_error' if problem_type == 'regression' else 'accuracy'
                    cv_scores = model_selection.cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    
                    if problem_type == 'classification':
                        accuracy = metrics.accuracy_score(y_test, y_pred)
                        precision = metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = metrics.f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        
                        # ROC-AUC for binary classification
                        if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
                            try:
                                y_proba = model.predict_proba(X_test)[:, 1]
                                roc_auc = metrics.roc_auc_score(y_test, y_proba)
                            except:
                                roc_auc = None
                        else:
                            roc_auc = None
                        
                        model_scores[name] = {
                            'cv_score': float(cv_scores.mean()),
                            'cv_std': float(cv_scores.std()),
                            'test_accuracy': float(accuracy),
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1_score': float(f1)
                        }
                        
                        if roc_auc is not None:
                            model_scores[name]['roc_auc'] = float(roc_auc)
                        
                        if accuracy > best_score:
                            best_score = accuracy
                            best_model_info = (name, model, model_scores[name])
                    
                    else:  # regression
                        r2 = metrics.r2_score(y_test, y_pred)
                        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                        mae = metrics.mean_absolute_error(y_test, y_pred)
                        
                        model_scores[name] = {
                            'cv_score': float(-cv_scores.mean()),
                            'cv_std': float(cv_scores.std()),
                            'test_r2': float(r2),
                            'rmse': float(rmse),
                            'mae': float(mae)
                        }
                        
                        if r2 > best_score:
                            best_score = r2
                            best_model_info = (name, model, model_scores[name])
                    
                    logger.info(f"Model {name} trained successfully")
                    
                except Exception as e:
                    logger.warning(f"Model {name} training failed: {e}")
                    continue
            
            automl_results["model_results"] = model_scores
            
            # Select best model
            if best_model_info:
                best_name, best_model, best_scores = best_model_info
                automl_results["best_model"] = {
                    'name': best_name,
                    'scores': best_scores
                }
                
                # Feature importance
                if hasattr(best_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': X.columns,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    # Use safe JSON serialization
                    automl_results["feature_importance"] = safe_json_serialize(importance_df.head(20))
                    
                    # Plot feature importance
                    plt.figure(figsize=(10, 6))
                    top_features = importance_df.head(10)
                    plt.barh(top_features['feature'], top_features['importance'])
                    plt.title(f'Top 10 Feature Importance - {best_name}')
                    plt.xlabel('Importance')
                    plt.tight_layout()
                    plot_path = save_plot_as_image(operation_id, "feature_importance")
                    if plot_path:
                        automl_results["plots_generated"].append(plot_path)
                
                # Save best model
                import joblib
                model_file = os.path.join(MODELS_DIR, f"best_model_{operation_id}.pkl")
                joblib.dump(best_model, model_file)
                
                # Register model
                tracker.register_model(f"model_{operation_id}", {
                    'name': best_name,
                    'type': problem_type,
                    'scores': best_scores,
                    'file_path': model_file
                })
        
        elif problem_type == 'clustering':
            # Unsupervised learning
            X = processed_df.select_dtypes(include=[np.number])
            if X.empty:
                return json.dumps({"error": "No numeric columns found for clustering", "operation_id": operation_id})
            
            # Feature scaling
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal number of clusters
            optimal_k, cluster_analysis = get_optimal_cluster_number(X_scaled, max_clusters=10)
            automl_results["cluster_analysis"] = cluster_analysis
            
            # Create clustering models
            clustering_models = create_advanced_models('clustering')
            
            # Remove advanced models if not requested
            if not include_advanced:
                advanced_clustering = ['HDBSCAN', 'Spectral', 'OPTICS']
                for name in advanced_clustering:
                    clustering_models.pop(name, None)
            
            # Set optimal cluster number for applicable models
            for name, model in clustering_models.items():
                if name in ['K-Means', 'Agglomerative', 'Spectral'] and hasattr(model, 'set_params'):
                    model.set_params(n_clusters=optimal_k)
                elif name == 'Gaussian Mixture' and hasattr(model, 'set_params'):
                    model.set_params(n_components=optimal_k)
            
            clustering_results = {}
            best_silhouette = -1
            best_clustering_info = None
            
            for name, model in clustering_models.items():
                try:
                    logger.info(f"Running {name} clustering...")
                    
                    if hasattr(model, 'fit_predict'):
                        cluster_labels = model.fit_predict(X_scaled)
                    else:
                        model.fit(X_scaled)
                        if hasattr(model, 'labels_'):
                            cluster_labels = model.labels_
                        else:
                            cluster_labels = model.predict(X_scaled)
                    
                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    
                    if n_clusters > 1:
                        silhouette = metrics.silhouette_score(X_scaled, cluster_labels)
                        clustering_results[name] = {
                            'silhouette_score': float(silhouette),
                            'n_clusters': int(n_clusters),
                            'inertia': float(model.inertia_) if hasattr(model, 'inertia_') else None
                        }
                        
                        if silhouette > best_silhouette:
                            best_silhouette = silhouette
                            best_clustering_info = (name, model, clustering_results[name], cluster_labels)
                    
                except Exception as e:
                    logger.warning(f"Clustering model {name} failed: {e}")
            
            automl_results["model_results"] = clustering_results
            
            # Save best clustering model
            if best_clustering_info:
                best_name, best_model, best_scores, best_labels = best_clustering_info
                automl_results["best_model"] = {
                    'name': best_name,
                    'scores': best_scores
                }
                
                # Create cluster visualization
                if X_scaled.shape[1] >= 2:
                    plt.figure(figsize=(10, 8))
                    
                    # Use PCA for visualization if more than 2 dimensions
                    if X_scaled.shape[1] > 2:
                        pca = decomposition.PCA(n_components=2)
                        X_viz = pca.fit_transform(X_scaled)
                        plt.scatter(X_viz[:, 0], X_viz[:, 1], c=best_labels, cmap='viridis', alpha=0.6)
                        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                    else:
                        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=best_labels, cmap='viridis', alpha=0.6)
                        plt.xlabel('Feature 1')
                        plt.ylabel('Feature 2')
                    
                    plt.title(f'Clustering Results - {best_name}')
                    plt.colorbar(label='Cluster')
                    plt.tight_layout()
                    
                    plot_path = save_plot_as_image(operation_id, "clustering_results")
                    if plot_path:
                        automl_results["plots_generated"].append(plot_path)
                
                # Save model
                import joblib
                model_file = os.path.join(MODELS_DIR, f"best_clustering_model_{operation_id}.pkl")
                joblib.dump(best_model, model_file)
                
                tracker.register_model(f"clustering_{operation_id}", {
                    'name': best_name,
                    'type': 'clustering',
                    'scores': best_scores,
                    'file_path': model_file
                })
        
        # Performance summary
        if automl_results["model_results"]:
            if problem_type == 'classification':
                key_metric = 'test_accuracy'
                metric_name = 'Accuracy'
            elif problem_type == 'regression':
                key_metric = 'test_r2'
                metric_name = 'R² Score'
            else:  # clustering
                key_metric = 'silhouette_score'
                metric_name = 'Silhouette Score'
            
            # Sort models by performance
            sorted_models = sorted(
                automl_results["model_results"].items(),
                key=lambda x: x[1].get(key_metric, -np.inf),
                reverse=True
            )
            
            automl_results["performance_summary"] = {
                "ranking": [
                    {
                        "rank": i + 1,
                        "model": name,
                        metric_name: scores.get(key_metric, 'N/A')
                    }
                    for i, (name, scores) in enumerate(sorted_models)
                ],
                "best_model": sorted_models[0][0] if sorted_models else None,
                "key_metric": metric_name
            }
        
        # Generate recommendations
        recommendations = []
        
        # Add sampling-related recommendations first
        if sampling_info and sampling_info.get("applied", False):
            recommendations.append(f"🤖 Model Training: Used {sampling_info.get('method', 'unknown')} sampled data ({len(df):,} from {sampling_info.get('original_rows', 'unknown'):,} rows)")
            quality = sampling_info.get('quality_check', {}).get('overall_quality', 'unknown')
            if quality == 'excellent':
                recommendations.append("✅ Sample quality: Excellent - model should be highly representative")
            elif quality == 'good':
                recommendations.append("✅ Sample quality: Good - model performance should be reliable")
            elif quality == 'fair':
                recommendations.append("⚠️ Sample quality: Fair - validate model on full dataset before production")
            
            recommendations.append("🔄 For production deployment, consider retraining on full dataset")
            recommendations.append("📊 Validate model performance on full dataset before final deployment")
        else:
            recommendations.append("✅ Model trained on complete dataset")
        
        # Model-specific recommendations
        if automl_results.get("best_model"):
            best_name = automl_results["best_model"]["name"]
            
            if best_name in ['XGBoost', 'LightGBM', 'CatBoost']:
                recommendations.append(f"⚡ {best_name} achieved best performance - gradient boosting excelled")
                recommendations.append("🔧 Consider hyperparameter tuning with Optuna for further improvement")
            elif 'Gaussian Process' in best_name:
                recommendations.append("🔬 Gaussian Process selected - uncertainty quantification available")
            elif 'Random Forest' in best_name:
                recommendations.append("🌲 Random Forest selected - robust to overfitting")
            elif 'SVM' in best_name or 'SVR' in best_name:
                recommendations.append("🎯 SVM selected - consider kernel tuning for better performance")
        
        # Problem-specific recommendations
        if problem_type in ['classification', 'regression']:
            recommendations.append("🔧 Consider feature engineering to improve model performance")
            recommendations.append("⚙️ Try hyperparameter tuning for better results")
            if automl_results.get("best_model"):
                best_score_key = 'test_accuracy' if problem_type == 'classification' else 'test_r2'
                best_score = automl_results["best_model"]["scores"].get(best_score_key, 0)
                if best_score < 0.7:
                    if sampling_info and sampling_info.get("applied", False):
                        recommendations.append("📈 Low performance detected - try with full dataset or improve data quality")
                    else:
                        recommendations.append("📈 Model performance is low. Consider data quality improvement")
                elif best_score > 0.9:
                    recommendations.append("🎯 High performance achieved. Validate on new data to avoid overfitting")
        else:  # clustering
            recommendations.append("🔍 Analyze cluster characteristics for business insights")
            recommendations.append("🔄 Consider different clustering algorithms for comparison")
            if cluster_analysis.get("optimal_clusters"):
                recommendations.append(f"📊 Optimal cluster count: {cluster_analysis['optimal_clusters']} (by silhouette analysis)")
        
        # Advanced feature recommendations
        if include_advanced:
            recommendations.append("🚀 Advanced models included - state-of-the-art performance possible")
        else:
            recommendations.append("💡 Enable include_advanced=True to access XGBoost, LightGBM, CatBoost")
        
        if SHAP_AVAILABLE and problem_type in ['classification', 'regression']:
            recommendations.append("🔍 SHAP values available for model interpretation")
        
        automl_results["recommendations"] = recommendations
        
        # Save AutoML report
        report_file = os.path.join(REPORTS_DIR, f"automl_report_{operation_id}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(automl_results, f, indent=2, ensure_ascii=False)
        
        # Log operation
        files_created = [report_file] + automl_results["plots_generated"]
        if code_path:
            files_created.append(code_path)
            
        tracker.log_operation(
            operation_id=operation_id,
            operation_type="auto_ml_pipeline",
            input_data={
                "dataset_id": dataset_id,
                "target_column": target_column,
                "test_size": test_size,
                "cv_folds": cv_folds,
                "include_advanced": include_advanced
            },
            output_data=automl_results,
            files_created=files_created,
            metrics={
                "models_trained": len(automl_results["model_results"]),
                "best_performance": best_score if 'best_score' in locals() else None
            }
        )
        
        logger.info(f"Advanced AutoML pipeline completed successfully: {operation_id}")
        return json.dumps(automl_results, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error in AutoML pipeline: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return json.dumps({"error": error_msg, "operation_id": operation_id})

@mcp.tool()
async def create_visualization(dataset_id: str, plot_type: str, 
                              x_column: str = None, y_column: str = None,
                              hue_column: str = None, title: str = None,
                              operation_id: str = None) -> str:
    """
    Create advanced data visualizations and save as images.
    
    Args:
        dataset_id (str): ID of the registered dataset
        plot_type (str): Type of plot (histogram, scatter, boxplot, heatmap, pairplot, violin, etc.)
        x_column (str): X-axis column name
        y_column (str): Y-axis column name
        hue_column (str): Column for color grouping
        title (str): Plot title
        operation_id (str): Optional operation ID for tracking
        
    Returns:
        str: Visualization result with image path
    """
    if operation_id is None:
        operation_id = f"viz_{uuid.uuid4().hex[:8]}"
    
    try:
        # Ensure sandbox structure exists
        if not ensure_sandbox_structure():
            error_msg = "Failed to ensure sandbox directory structure"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        logger.info(f"Creating visualization: {plot_type} for dataset {dataset_id}")
        
        # Load dataset
        dataset_file = os.path.join(DATASETS_DIR, f"{dataset_id}.csv")
        if not os.path.exists(dataset_file):
            error_msg = f"Dataset {dataset_id} not found"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        df = pd.read_csv(dataset_file)
        
        # Set default title
        if title is None:
            title = f"{plot_type.title()} Plot"
        
        # Generate visualization code
        viz_code = generate_visualization_code(dataset_id, plot_type, x_column, y_column, title)
        code_path = save_code_as_file(operation_id, viz_code, "visualization")
        
        plt.figure(figsize=(10, 6))
        
        if plot_type.lower() == 'histogram':
            if x_column and x_column in df.columns:
                df[x_column].hist(bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel(x_column)
                plt.ylabel('Frequency')
            else:
                return json.dumps({"error": "x_column required for histogram", "operation_id": operation_id})
                
        elif plot_type.lower() == 'scatter':
            if x_column and y_column and x_column in df.columns and y_column in df.columns:
                if hue_column and hue_column in df.columns:
                    for category in df[hue_column].unique():
                        mask = df[hue_column] == category
                        plt.scatter(df[mask][x_column], df[mask][y_column], label=category, alpha=0.6)
                    plt.legend()
                else:
                    plt.scatter(df[x_column], df[y_column], alpha=0.6)
                plt.xlabel(x_column)
                plt.ylabel(y_column)
            else:
                return json.dumps({"error": "Both x_column and y_column required for scatter plot", "operation_id": operation_id})
                
        elif plot_type.lower() == 'boxplot':
            if x_column and x_column in df.columns:
                if y_column and y_column in df.columns:
                    df.boxplot(column=y_column, by=x_column)
                    plt.suptitle('')  # Remove automatic title
                else:
                    df.boxplot(column=x_column)
            else:
                return json.dumps({"error": "x_column required for boxplot", "operation_id": operation_id})
                
        elif plot_type.lower() == 'heatmap':
            numeric_cols = df.select_dtypes(include=['number'])
            if len(numeric_cols.columns) > 1:
                correlation_matrix = numeric_cols.corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            else:
                return json.dumps({"error": "Need at least 2 numeric columns for heatmap", "operation_id": operation_id})
                
        elif plot_type.lower() == 'bar':
            if x_column and x_column in df.columns:
                if y_column and y_column in df.columns:
                    df.groupby(x_column)[y_column].mean().plot(kind='bar')
                    plt.ylabel(f'Mean {y_column}')
                else:
                    df[x_column].value_counts().plot(kind='bar')
                    plt.ylabel('Count')
                plt.xlabel(x_column)
                plt.xticks(rotation=45)
            else:
                return json.dumps({"error": "x_column required for bar plot", "operation_id": operation_id})
        
        elif plot_type.lower() == 'pairplot':
            # Use seaborn for pairplot
            numeric_cols = df.select_dtypes(include=['number']).columns[:6]  # Limit to 6 columns
            if len(numeric_cols) >= 2:
                plt.close()  # Close current figure
                if hue_column and hue_column in df.columns:
                    pairplot = sns.pairplot(df[list(numeric_cols) + [hue_column]], hue=hue_column)
                else:
                    pairplot = sns.pairplot(df[numeric_cols])
                pairplot.fig.suptitle(title, y=1.02)
            else:
                return json.dumps({"error": "Need at least 2 numeric columns for pairplot", "operation_id": operation_id})
        
        elif plot_type.lower() == 'violin':
            if x_column and y_column:
                if x_column in df.columns and y_column in df.columns:
                    if df[y_column].dtype.kind in 'biufc':
                        sns.violinplot(x=x_column, y=y_column, data=df)
                    else:
                        sns.violinplot(x=y_column, y=x_column, data=df)
                    plt.xticks(rotation=45)
                else:
                    return json.dumps({"error": "Both x_column and y_column required for violin plot", "operation_id": operation_id})
            else:
                return json.dumps({"error": "Both x_column and y_column required for violin plot", "operation_id": operation_id})
        
        elif plot_type.lower() == 'distribution':
            if x_column and x_column in df.columns:
                plt.figure(figsize=(12, 5))
                
                # Subplot 1: Histogram with KDE
                plt.subplot(1, 2, 1)
                df[x_column].hist(bins=30, alpha=0.7, density=True, edgecolor='black')
                df[x_column].plot.kde()
                plt.xlabel(x_column)
                plt.ylabel('Density')
                plt.title(f'Distribution of {x_column}')
                
                # Subplot 2: Q-Q plot
                plt.subplot(1, 2, 2)
                from scipy import stats
                stats.probplot(df[x_column].dropna(), dist="norm", plot=plt)
                plt.title(f'Q-Q Plot of {x_column}')
            else:
                return json.dumps({"error": "x_column required for distribution plot", "operation_id": operation_id})
        
        else:
            return json.dumps({"error": f"Unsupported plot type: {plot_type}", "operation_id": operation_id})
        
        if plot_type.lower() != 'pairplot':
            plt.title(title)
        plt.tight_layout()
        
        # Save plot
        plot_path = save_plot_as_image(operation_id, plot_type)
        
        if not plot_path:
            return json.dumps({"error": "Failed to save plot", "operation_id": operation_id})
        
        # Prepare result
        viz_result = {
            "operation_id": operation_id,
            "dataset_id": dataset_id,
            "plot_type": plot_type,
            "plot_path": plot_path,
            "x_column": x_column,
            "y_column": y_column,
            "hue_column": hue_column,
            "title": title,
            "generated_code": code_path if code_path else None,
            "status": "success"
        }
        
        # Log operation
        files_created = [plot_path]
        if code_path:
            files_created.append(code_path)
            
        tracker.log_operation(
            operation_id=operation_id,
            operation_type="create_visualization",
            input_data={
                "dataset_id": dataset_id,
                "plot_type": plot_type,
                "x_column": x_column,
                "y_column": y_column,
                "hue_column": hue_column,
                "title": title
            },
            output_data=viz_result,
            files_created=files_created
        )
        
        logger.info(f"Visualization created successfully: {plot_path}")
        return json.dumps(viz_result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error creating visualization: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        plt.close()  # Ensure plot is closed on error
        return json.dumps({"error": error_msg, "operation_id": operation_id})

@mcp.tool()
async def get_operation_details(operation_id: str) -> str:
    """
    Get detailed information about a specific operation.
    
    Args:
        operation_id (str): ID of the operation to retrieve
        
    Returns:
        str: Operation details including inputs, outputs, and files created
    """
    try:
        # Ensure sandbox structure exists
        if not ensure_sandbox_structure():
            error_msg = "Failed to ensure sandbox directory structure"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        operation = tracker.get_operation(operation_id)
        
        if operation is None:
            # Try to load from log file
            log_file = os.path.join(LOGS_DIR, f'operation_{operation_id}.json')
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    operation = json.load(f)
            else:
                error_msg = f"Operation {operation_id} not found"
                logger.error(error_msg)
                return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        # Add file existence check
        if 'files_created' in operation:
            file_status = {}
            for file_path in operation['files_created']:
                file_status[file_path] = os.path.exists(file_path)
            operation['file_status'] = file_status
        
        logger.info(f"Retrieved operation details: {operation_id}")
        return json.dumps(operation, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error retrieving operation {operation_id}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg, "operation_id": operation_id})

@mcp.tool()
async def list_available_datasets() -> str:
    """
    List all available datasets in the sandbox.
    
    Returns:
        str: List of available datasets with basic information
    """
    try:
        # Ensure sandbox structure exists
        if not ensure_sandbox_structure():
            error_msg = "Failed to ensure sandbox directory structure"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})
        
        datasets = []
        
        # List CSV files in datasets directory
        for file_path in Path(DATASETS_DIR).glob("*.csv"):
            try:
                # Try to get basic info about the dataset
                df = pd.read_csv(file_path, nrows=5)  # Read only first 5 rows for speed
                dataset_info = {
                    "dataset_id": file_path.stem,
                    "file_path": str(file_path),
                    "shape": df.shape,
                    "columns": df.columns.tolist()[:10],  # Limit to first 10 columns
                    "size_mb": file_path.stat().st_size / 1024 / 1024,
                    "created_time": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
                }
                
                # Check if this dataset has been registered
                if file_path.stem in tracker.datasets:
                    dataset_info["registered"] = True
                    dataset_info["metadata"] = tracker.datasets[file_path.stem]
                else:
                    dataset_info["registered"] = False
                
                datasets.append(dataset_info)
            except Exception as e:
                logger.warning(f"Could not read dataset {file_path}: {e}")
        
        # Sort by creation time (newest first)
        datasets.sort(key=lambda x: x['created_time'], reverse=True)
        
        result = {
            "available_datasets": datasets,
            "total_count": len(datasets),
            "datasets_directory": DATASETS_DIR
        }
        
        logger.info(f"Listed {len(datasets)} available datasets")
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error listing datasets: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

@mcp.tool()
async def get_environment_info() -> str:
    """
    Get information about the data science environment and available packages.
    
    Returns:
        str: Environment information including package availability
    """
    try:
        # Ensure sandbox structure exists
        if not ensure_sandbox_structure():
            error_msg = "Failed to ensure sandbox directory structure"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})
        
        env_info = {
            "sandbox_directory": SANDBOX_DIR,
            "subdirectories": {
                "datasets": DATASETS_DIR,
                "plots": PLOTS_DIR,
                "models": MODELS_DIR,
                "reports": REPORTS_DIR,
                "logs": LOGS_DIR,
                "generated_code": CODE_DIR
            },
            "core_packages": {
                "pandas": True,
                "numpy": True,
                "matplotlib": True,
                "seaborn": True,
                "scikit_learn": SKLEARN_AVAILABLE,
                "plotly": PLOTLY_AVAILABLE
            },
            "ml_packages": {
                "tensorflow": TF_AVAILABLE,
                "xgboost": XGB_AVAILABLE,
                "lightgbm": LGB_AVAILABLE,
                "catboost": CATBOOST_AVAILABLE,
                "autokeras": AUTOKERAS_AVAILABLE,
                "keras_tuner": KERAS_TUNER_AVAILABLE
            },
            "advanced_packages": {
                "statsmodels": STATSMODELS_AVAILABLE,
                "shap": SHAP_AVAILABLE,
                "umap": UMAP_AVAILABLE,
                "hdbscan": HDBSCAN_AVAILABLE,
                "imbalanced_learn": IMBLEARN_AVAILABLE,
                "optuna": OPTUNA_AVAILABLE
            },
            "supported_features": [
                "Smart data loading with adaptive sampling",
                "Advanced AutoML with problem type detection",
                "Deep learning with AutoKeras integration",
                "High-performance algorithms (XGBoost, LightGBM, CatBoost)",
                "Gaussian Processes and advanced algorithms",
                "Clustering with silhouette optimization",
                "Advanced dimensionality reduction (t-SNE, UMAP)",
                "Model interpretation with SHAP",
                "Interactive visualizations with Plotly",
                "Complete Python code generation",
                "Comprehensive markdown reports",
                "File upload and management"
            ],
            "problem_types_supported": [
                "classification",
                "regression",
                "clustering",
                "anomaly_detection",
                "timeseries",
                "timeseries_forecasting",
                "text_classification",
                "text_analysis",
                "image_classification",
                "image_regression",
                "image_analysis"
            ],
            "visualization_types": [
                "histogram",
                "scatter",
                "boxplot",
                "heatmap",
                "bar",
                "pairplot",
                "violin",
                "distribution"
            ]
        }
        
        logger.info("Retrieved environment information")
        return json.dumps(env_info, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error getting environment info: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

@mcp.tool()
async def list_generated_code() -> str:
    """
    List all generated Python code files.
    
    Returns:
        str: List of generated code files with basic information
    """
    try:
        # Ensure sandbox structure exists
        if not ensure_sandbox_structure():
            error_msg = "Failed to ensure sandbox directory structure"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})
        
        code_files = []
        
        # List Python files in code directory
        for file_path in Path(CODE_DIR).glob("*.py"):
            try:
                file_info = {
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "size_kb": file_path.stat().st_size / 1024,
                    "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "operation_id": file_path.stem.split('_')[0] if '_' in file_path.stem else "unknown",
                    "code_type": file_path.stem.split('_')[1] if '_' in file_path.stem else "unknown"
                }
                code_files.append(file_info)
            except Exception as e:
                logger.warning(f"Could not read code file {file_path}: {e}")
        
        # Sort by modification time (newest first)
        code_files.sort(key=lambda x: x['modified_time'], reverse=True)
        
        result = {
            "generated_code_files": code_files,
            "total_count": len(code_files),
            "code_directory": CODE_DIR,
            "description": "All generated code files are standalone Python scripts that can be run independently."
        }
        
        logger.info(f"Listed {len(code_files)} generated code files")
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error listing generated code: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

@mcp.tool()
async def upload_local_file(file_path: str, operation_id: str = None) -> str:
    """
    Upload and register a local file for data science analysis.
    
    Args:
        file_path (str): Full path to the local file to upload
        operation_id (str): Optional operation ID for tracking
        
    Returns:
        str: Upload result with file information and next steps
    """
    if operation_id is None:
        operation_id = f"upload_{uuid.uuid4().hex[:8]}"
    
    try:
        # Ensure sandbox structure exists
        if not ensure_sandbox_structure():
            error_msg = "Failed to ensure sandbox directory structure"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        logger.info(f"Uploading local file: {file_path}")
        
        # Check if source file exists
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        # Validate file
        file_ext = Path(file_path).suffix.lower()
        supported_extensions = {'.csv', '.xlsx', '.xls', '.json', '.parquet', '.txt', '.tsv'}
        
        if file_ext not in supported_extensions:
            error_msg = f"Unsupported file format: {file_ext}. Supported formats: {', '.join(supported_extensions)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        # Get file info
        file_stats = os.stat(file_path)
        file_size_mb = file_stats.st_size / (1024 * 1024)
        original_filename = os.path.basename(file_path)
        
        # Generate safe filename for destination
        clean_name = "".join(c for c in Path(original_filename).stem if c.isalnum() or c in ('-', '_', ' ')).strip()
        if len(clean_name) > 50:
            clean_name = clean_name[:50]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{clean_name}_{timestamp}{file_ext}"
        
        # Ensure uniqueness
        counter = 1
        base_filename = safe_filename
        destination_path = os.path.join(DATASETS_DIR, safe_filename)
        while os.path.exists(destination_path):
            name_part = Path(base_filename).stem
            ext_part = Path(base_filename).suffix
            safe_filename = f"{name_part}_{counter}{ext_part}"
            destination_path = os.path.join(DATASETS_DIR, safe_filename)
            counter += 1
        
        # Copy file to datasets directory
        try:
            shutil.copy2(file_path, destination_path)
            logger.info(f"File copied: {file_path} -> {destination_path}")
        except Exception as e:
            error_msg = f"Failed to copy file: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        # Prepare response
        upload_info = {
            "operation_id": operation_id,
            "status": "success",
            "message": "File uploaded successfully",
            "file_info": {
                "original_path": file_path,
                "original_filename": original_filename,
                "saved_filename": safe_filename,
                "destination_path": destination_path,
                "file_size_mb": round(file_size_mb, 2),
                "file_extension": file_ext,
                "uploaded_at": datetime.now().isoformat()
            },
            "warnings": [],
            "next_steps": [
                f"File uploaded successfully: {safe_filename}",
                "Use load_dataset tool to analyze the data",
                f"Example: load_dataset('{destination_path}')",
                "Large files will be automatically sampled for performance"
            ]
        }
        
        # Add warnings for large files
        if file_size_mb > 100:
            upload_info["warnings"].append(f"Large file ({file_size_mb:.1f}MB). Automatic sampling will be applied during loading.")
        
        # Try to get basic data preview
        try:
            if file_ext == '.csv':
                df_preview = pd.read_csv(destination_path, nrows=5)
            elif file_ext in ['.xlsx', '.xls']:
                df_preview = pd.read_excel(destination_path, nrows=5)
            elif file_ext == '.json':
                df_preview = pd.read_json(destination_path).head(5)
            elif file_ext == '.parquet':
                df_preview = pd.read_parquet(destination_path).head(5)
            else:
                df_preview = None
            
            if df_preview is not None:
                upload_info["data_preview"] = {
                    "columns": df_preview.columns.tolist(),
                    "column_count": len(df_preview.columns),
                    "sample_rows": len(df_preview),
                    "dtypes": {col: str(dtype) for col, dtype in df_preview.dtypes.items()}
                }
        except Exception as e:
            upload_info["data_preview"] = {"error": f"Failed to preview data: {str(e)}"}
        
        # Log upload operation
        tracker.log_operation(
            operation_id=operation_id,
            operation_type="upload_local_file",
            input_data={
                "original_path": file_path,
                "original_filename": original_filename,
                "file_size_mb": file_size_mb
            },
            output_data=upload_info,
            files_created=[destination_path]
        )
        
        logger.info(f"Local file upload completed: {safe_filename} ({file_size_mb:.2f}MB)")
        return json.dumps(upload_info, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error uploading file: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return json.dumps({"error": error_msg, "operation_id": operation_id})

@mcp.tool()
async def copy_file_to_sandbox(source_path: str, target_filename: str = None, operation_id: str = None) -> str:
    """
    Copy a file from any location to the sandbox datasets directory.
    
    Args:
        source_path (str): Full path to the source file
        target_filename (str): Optional custom filename for the copied file
        operation_id (str): Optional operation ID for tracking
        
    Returns:
        str: Copy operation result
    """
    if operation_id is None:
        operation_id = f"copy_{uuid.uuid4().hex[:8]}"
    
    try:
        # Ensure sandbox structure exists
        if not ensure_sandbox_structure():
            error_msg = "Failed to ensure sandbox directory structure"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        # Check source file
        if not os.path.exists(source_path):
            error_msg = f"Source file not found: {source_path}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "operation_id": operation_id})
        
        # Determine target filename
        if target_filename:
            # Use provided filename
            safe_filename = target_filename
        else:
            # Generate safe filename
            original_filename = os.path.basename(source_path)
            clean_name = "".join(c for c in Path(original_filename).stem if c.isalnum() or c in ('-', '_', ' ')).strip()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_ext = Path(original_filename).suffix.lower()
            safe_filename = f"{clean_name}_{timestamp}{file_ext}"
        
        destination_path = os.path.join(DATASETS_DIR, safe_filename)
        
        # Ensure no conflicts
        counter = 1
        base_path = destination_path
        while os.path.exists(destination_path):
            name_part = Path(base_path).stem
            ext_part = Path(base_path).suffix
            destination_path = os.path.join(DATASETS_DIR, f"{name_part}_{counter}{ext_part}")
            counter += 1
        
        # Copy file
        shutil.copy2(source_path, destination_path)
        
        # Get file info
        file_stats = os.stat(destination_path)
        file_size_mb = file_stats.st_size / (1024 * 1024)
        
        result = {
            "operation_id": operation_id,
            "status": "success",
            "message": "File copied successfully",
            "source_path": source_path,
            "destination_path": destination_path,
            "filename": os.path.basename(destination_path),
            "file_size_mb": round(file_size_mb, 2),
            "copied_at": datetime.now().isoformat()
        }
        
        # Log operation
        tracker.log_operation(
            operation_id=operation_id,
            operation_type="copy_file_to_sandbox",
            input_data={"source_path": source_path, "target_filename": target_filename},
            output_data=result,
            files_created=[destination_path]
        )
        
        logger.info(f"File copied: {source_path} -> {destination_path}")
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error copying file: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg, "operation_id": operation_id})

@mcp.tool()
async def list_uploaded_files() -> str:
    """
    List all uploaded files in the datasets directory.
    
    Returns:
        str: List of uploaded files with details
    """
    try:
        # Ensure sandbox structure exists
        if not ensure_sandbox_structure():
            error_msg = "Failed to ensure sandbox directory structure"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})
        
        uploaded_files = []
        
        # List all files in datasets directory
        if os.path.exists(DATASETS_DIR):
            for file_path in Path(DATASETS_DIR).iterdir():
                if file_path.is_file():
                    try:
                        stats = file_path.stat()
                        file_info = {
                            "filename": file_path.name,
                            "file_path": str(file_path),
                            "size_mb": round(stats.st_size / (1024 * 1024), 2),
                            "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                            "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                            "extension": file_path.suffix.lower()
                        }
                        
                        # Try to get basic data info for data files
                        if file_path.suffix.lower() in ['.csv', '.xlsx', '.xls', '.json', '.parquet']:
                            try:
                                if file_path.suffix.lower() == '.csv':
                                    df_sample = pd.read_csv(file_path, nrows=5)
                                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                                    df_sample = pd.read_excel(file_path, nrows=5)
                                elif file_path.suffix.lower() == '.json':
                                    df_sample = pd.read_json(file_path).head(5)
                                elif file_path.suffix.lower() == '.parquet':
                                    df_sample = pd.read_parquet(file_path).head(5)
                                
                                file_info["data_preview"] = {
                                    "columns": df_sample.columns.tolist(),
                                    "column_count": len(df_sample.columns),
                                    "sample_rows": len(df_sample)
                                }
                            except Exception as e:
                                file_info["data_preview"] = {"error": f"Failed to preview: {str(e)}"}
                        
                        uploaded_files.append(file_info)
                    except Exception as e:
                        logger.warning(f"Failed to get info for file {file_path}: {e}")
        
        # Sort by modification time (newest first)
        uploaded_files.sort(key=lambda x: x['modified_at'], reverse=True)
        
        result = {
            "uploaded_files": uploaded_files,
            "total_count": len(uploaded_files),
            "datasets_directory": DATASETS_DIR,
            "upload_info": {
                "upload_methods": [
                    "upload_local_file(file_path) - Upload from local file path",
                    "copy_file_to_sandbox(source_path, target_filename) - Copy file with custom name"
                ],
                "supported_formats": [".csv", ".xlsx", ".xls", ".json", ".parquet", ".txt", ".tsv"],
                "note": "Uploaded files can be analyzed using the load_dataset tool"
            }
        }
        
        logger.info(f"Listed {len(uploaded_files)} uploaded files")
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error listing uploaded files: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

@mcp.tool()
async def get_upload_instructions() -> str:
    """
    Get detailed instructions for file upload functionality.
    
    Returns:
        str: Upload instructions and information
    """
    try:
        # Ensure sandbox structure exists
        if not ensure_sandbox_structure():
            error_msg = "Failed to ensure sandbox directory structure"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})
        
        instructions = {
            "upload_guide": {
                "title": "File Upload Guide",
                "description": "You can upload local files for data science analysis using the server's file management tools.",
                "methods": [
                    {
                        "method": "upload_local_file",
                        "description": "Upload a file from your local system to the sandbox",
                        "usage": "upload_local_file('/path/to/your/data.csv')",
                        "features": ["Automatic filename sanitization", "Duplicate prevention", "Data preview", "Size checking"]
                    },
                    {
                        "method": "copy_file_to_sandbox", 
                        "description": "Copy a file with optional custom naming",
                        "usage": "copy_file_to_sandbox('/path/to/source.csv', 'my_dataset.csv')",
                        "features": ["Custom naming", "Flexible copying", "Safe handling"]
                    }
                ]
            },
            "supported_formats": {
                "data_files": [".csv", ".xlsx", ".xls", ".json", ".parquet"],
                "text_files": [".txt", ".tsv"],
                "notes": "These formats are optimized for data analysis. Other formats can be copied but may require custom processing."
            },
            "workflow_steps": [
                "1. Prepare your local file (check supported formats)",
                "2. Use upload_local_file or copy_file_to_sandbox tool",
                "3. Note the destination_path from the result",
                "4. Use load_dataset tool with the destination_path",
                "5. Start analysis with perform_eda and auto_ml_pipeline"
            ],
            "automatic_features": {
                "large_file_handling": "Files >100MB will be automatically sampled during load_dataset",
                "file_validation": "Format checking and safety validation",
                "operation_tracking": "All uploads are logged for tracking",
                "safe_naming": "Special characters removed and timestamps added"
            },
            "usage_examples": {
                "basic_upload": """
# 1. Upload local file
result = await upload_local_file('C:/Users/user/data/sales_data.csv')

# 2. Start analysis with uploaded file
await load_dataset(result['file_info']['destination_path'], target_column='target')
await perform_eda('dataset_abc123')
await auto_ml_pipeline('dataset_abc123', 'target')
                """,
                "custom_naming": """
# Copy with custom name
await copy_file_to_sandbox('C:/data/raw_data.xlsx', 'clean_dataset.xlsx')
await load_dataset('./sandbox/datasets/clean_dataset.xlsx')
                """,
                "large_file_workflow": """
# Large file handling (automatic sampling)
await upload_local_file('C:/bigdata/huge_dataset.csv')  # 500MB file
await load_dataset('./sandbox/datasets/huge_dataset_20241207_143052.csv', target_column='label')
# -> Automatically sampled for performance
                """
            },
            "destination_info": {
                "sandbox_directory": SANDBOX_DIR,
                "datasets_directory": DATASETS_DIR,
                "naming_convention": "originalname_YYYYMMDD_HHMMSS.extension",
                "duplicate_handling": "Automatic numbering for duplicates"
            }
        }
        
        logger.info("Provided file upload instructions")
        return json.dumps(instructions, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error generating upload instructions: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

@mcp.tool()
async def generate_comprehensive_report(
    operations: List[str] = None,
    report_title: str = "Advanced Data Science Analysis Report",
    dataset_name: str = "Dataset",
    embed_figures: bool = True,
    add_table_of_contents: bool = True,
    use_llm: bool = True
) -> str:
    """
    Generate a comprehensive analysis report from multiple operations.
    
    Args:
        operations: List of operation IDs to include in the report (None = use all recent operations)
        report_title: Title for the report
        dataset_name: Name of the analyzed dataset
        embed_figures: Whether to embed images in the report
        add_table_of_contents: Whether to include a table of contents
        use_llm: Whether to use LLM for enhancing conclusions
        
    Returns:
        str: JSON with report generation result and file path
    """
    try:
        # If no operations specified, find recent operations
        if operations is None:
            operations = []
            
            # Get all operations from tracker
            all_ops = list(tracker.operations.keys())
            
            # Also check log files
            if os.path.exists(LOGS_DIR):
                for log_file in glob.glob(os.path.join(LOGS_DIR, "operation_*.json")):
                    op_id = os.path.basename(log_file).replace('operation_', '').replace('.json', '')
                    if op_id not in all_ops:
                        all_ops.append(op_id)
            
            # Sort by timestamp (assuming operation IDs contain timestamps)
            all_ops.sort(reverse=True)
            
            # Find the most recent load_dataset, perform_eda, and auto_ml_pipeline operations
            for op_id in all_ops:
                op = tracker.get_operation(op_id)
                if not op:
                    # Try loading from file
                    log_file = os.path.join(LOGS_DIR, f'operation_{op_id}.json')
                    if os.path.exists(log_file):
                        try:
                            with open(log_file, 'r') as f:
                                op = json.load(f)
                        except:
                            continue
                
                if op:
                    op_type = op.get('type', '')
                    if op_type == 'load_dataset' and 'load_dataset' not in [tracker.get_operation(oid).get('type') for oid in operations if tracker.get_operation(oid)]:
                        operations.append(op_id)
                    elif op_type == 'perform_eda' and 'perform_eda' not in [tracker.get_operation(oid).get('type') for oid in operations if tracker.get_operation(oid)]:
                        operations.append(op_id)
                    elif op_type == 'auto_ml_pipeline' and 'auto_ml_pipeline' not in [tracker.get_operation(oid).get('type') for oid in operations if tracker.get_operation(oid)]:
                        operations.append(op_id)
                
                # Stop if we have all three types
                if len(operations) >= 3:
                    break
        
        if not operations:
            return json.dumps({
                "status": "error",
                "error": "No operations found to generate report. Please run load_dataset, perform_eda, and/or auto_ml_pipeline first."
            }, indent=2, ensure_ascii=False)
        
        # Call the main report generation function
        result = create_comprehensive_report(
            operations=operations,
            report_title=report_title,
            dataset_name=dataset_name,
            embed_figures=embed_figures,
            add_table_of_contents=add_table_of_contents,
            use_llm=use_llm
        )
        
        return result
        
    except Exception as e:
        error_msg = f"Error generating comprehensive report: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return json.dumps({
            "status": "error",
            "error": error_msg
        }, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    logger.info("Starting Advanced Data Science MCP Server...")
    logger.info(f"Sandbox directory: {SANDBOX_DIR}")
    logger.info(f"Server port: {SERVER_PORT}")
    logger.info("Package availability:")
    logger.info(f"  Core ML: sklearn={SKLEARN_AVAILABLE}, plotly={PLOTLY_AVAILABLE}")
    logger.info(f"  Deep Learning: tensorflow={TF_AVAILABLE}, autokeras={AUTOKERAS_AVAILABLE}")
    logger.info(f"  Boosting: xgboost={XGB_AVAILABLE}, lightgbm={LGB_AVAILABLE}, catboost={CATBOOST_AVAILABLE}")
    logger.info(f"  Advanced: shap={SHAP_AVAILABLE}, umap={UMAP_AVAILABLE}, hdbscan={HDBSCAN_AVAILABLE}")
    logger.info("Features: Code generation enabled, File upload enabled")
    logger.info("Available tools: upload_local_file, copy_file_to_sandbox")
    
    try:
        # Get the SSE app
        app = mcp.sse_app()
        
        # Configure uvicorn with better settings for stability
        config = uvicorn.Config(
            app, 
            host="0.0.0.0", 
            port=SERVER_PORT,
            log_level="info",
            access_log=True,
            # Better timeout settings
            timeout_keep_alive=30,
            timeout_graceful_shutdown=10,
            # Limit concurrent connections to prevent resource issues
            limit_concurrency=100,
            limit_max_requests=1000,
        )
        
        server = uvicorn.Server(config)
        logger.info(f"Server starting at http://0.0.0.0:{SERVER_PORT}")
        logger.info("Use 'health_check' tool to verify server status")
        server.run()
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)