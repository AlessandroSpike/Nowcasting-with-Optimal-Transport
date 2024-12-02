This repository contains code for implementing a novel nowcasting framework that leverages Optimal Transport (OT) theory to enhance short-term forecasting. The methodology includes both a non-parametric approach for imputing missing values using the Sinkhorn divergence and a parametric integration into Dynamic Factor Models (DFM) via an Expectation-Maximization (EM) algorithm.

RealNonParamNEW.m
A MATLAB implementation of a nowcasting framework using Sinkhorn divergence-based imputation for macroeconomic datasets. The script handles missing data and performs nowcasting across various scenarios, including COVID-19 recession periods.

Core Functions:
- Data preprocessing with standardization and normalization
- Sinkhorn divergence-based imputation with configurable parameters
- Support for alternative methods (KNN, Regression Trees)
- Iterative forecasting simulation
- Results collection and storage

Configuration options include adjustable Sinkhorn algorithm parameters and selective period handling. Outputs are saved in structured .mat files containing nowcasting results and iteration details.

RealNowcast_ISNEW.m
An integrated nowcasting framework combining Sinkhorn-based imputation, Dynamic Factor Model (DFM), and Autoregressive (AR) approaches. Designed for handling macroeconomic time series with missing values.

Key Components:
- Comprehensive data preparation with GDP variable handling
- Multiple model implementations:
  - Sinkhorn-based imputation with dynamic regularization
  - DFM with EM algorithm and Kalman filtering
  - AR model with automatic order selection
- Iterative nowcasting simulation
- Performance evaluation using RMSE
- Structured output storage

RealNowcast_MIX.m
A hybrid nowcasting framework combining Sinkhorn imputation and DFM with specialized outlier handling. 

Main Features:
- Flexible data preprocessing with COVID-19 period handling
- DFM configuration with customizable factors and lags
- Outlier detection using interquartile range method
- Adaptive method selection based on outlier presence
- Performance tracking and result storage

SimulazioniBasic.m
A simulation framework comparing DFM and Sinkhorn-based nowcasting approaches.

Components:
- Configurable simulation parameters
- Synthetic data generation with factor model
- Implementation of DFM and Sinkhorn methods
- Comprehensive evaluation metrics (RMSE, R²)
- Result logging and storage

SimulazioniIS.m
Extended simulation framework incorporating time-dependent idiosyncratic dynamics for comparing DFM and Sinkhorn-based methods.

Features:
- Advanced data generation with dynamic factors
- Idiosyncratic dynamics modeling
- Comparative implementation of both methods
- Detailed performance metrics
- Comprehensive result storage

SimulazioniNonParam.m
Evaluation framework for comparing KNN, regression trees, and Sinkhorn-based imputation methods.

Includes:
- Flexible simulation parameters
- Multiple imputation method implementations
- Configurable method hyperparameters
- Performance evaluation across varying missing data scenarios
- Result collection and analysis tools

The files with the *PARALLEL* suffix employ `parfor` loops to accelerate computational time by parallelizing the tasks across multiple workers. Similarly, the files with the *EXPANDING* suffix also utilize parallel processing but with the added functionality of expanding windows, enabling more efficient handling of time-varying data by distributing the workload in parallel across dynamic window sizes. Both implementations significantly reduce processing time by leveraging parallel computing.

Remember to use `genpath` to generate the full path of subdirectories and `addpath` to include the necessary directories in your MATLAB environment for accessing all required functions and scripts.
