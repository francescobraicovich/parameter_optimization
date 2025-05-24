# Machine Learning Parameter Optimization

This repository demonstrates advanced hyperparameter optimization techniques for classification algorithms on a multi-class dataset. The project provides a comprehensive comparison between k-Nearest Neighbors (KNN) and Logistic Regression algorithms, along with ensemble methods and deep learning approaches.

## Project Overview

This project compares different machine learning algorithms and optimization strategies on a dataset with 1000 samples, each with 35 features. The classification task involves predicting one of three possible class labels.

###  Key Features

- Comprehensive exploratory data analysis
- Multiple hyperparameter optimization techniques:
  - Grid Search
  - Random Search
  - Optuna for Bayesian optimization
  - Scikit-Optimize Bayesian optimization
- Advanced model comparison methodologies
- Feature engineering and selection
- Ensemble learning with stacking classifier
- Neural network implementation with PyTorch

## Dataset

The dataset consists of:
- 1000 samples with 35 features
- 3 possible target classes (balanced distribution)
- Primarily numerical features with no missing values

## Repository Structure

- `main.ipynb`: Main notebook with all analyses, models and visualizations
- `data.csv`: Labeled dataset for training and testing
- `data_unlabeled.csv`: Unlabeled dataset for predictions
- `description.txt`: Problem description and requirements
- `best_model.pt`: Saved PyTorch neural network model
- `predictions_nn.csv`: Neural network predictions
- `predictions_stacking_clf_knn_logreg.csv`: Stacking classifier predictions

## Installation and Usage

1. Clone this repository
2. Install the required packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch optuna skopt umap-learn
```
3. Run the Jupyter notebook to see the full analysis