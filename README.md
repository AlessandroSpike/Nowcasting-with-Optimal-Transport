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

The files with the PARALLEL suffix employ `parfor` loops to accelerate computational time by parallelizing the tasks across multiple workers. Similarly, the files with the EXPANDING suffix also utilize parallel processing but with the added functionality of expanding windows, enabling more efficient handling of time-varying data by distributing the workload in parallel across dynamic window sizes. Both implementations significantly reduce processing time by leveraging parallel computing.

Functions Overview
The Function folder contains the core implementation of an advanced nowcasting methodology that integrates several sophisticated techniques for handling missing data and estimating dynamic system parameters. These MATLAB functions implement a comprehensive framework for data imputation, dynamic factor modeling, and state estimation.

OT_imputerCONDITIONAL.m
Implements the Optimal Transport (OT) imputation method with conditional optimization for missing data. This function leverages Sinkhorn divergence to iteratively impute missing values while preserving the data's distribution.

Core Features:  
- Conditional optimization with batch-based sampling  
- Sinkhorn divergence computation using entropy regularization  
- Gradient-based updates with RMSprop optimization  
- Convergence tracking through divergence monitoring  

Configurable parameters include batch size, learning rate, and regularization parameter. Outputs the final imputed matrix and divergence values for each iteration.

OT_imputer.m  
A streamlined OT imputation method compared to `OT_imputerCONDITIONAL`. This version initializes transport plans uniformly and generalizes the Sinkhorn updates, reducing computational complexity.

Key Differences:  
- Uniform initialization for transport plans (`mu` and `nu`)  
- Generalized gradient computations without conditional updates  
- Simplified divergence tracking compared to the conditional version  

Best suited for cases where conditional updates are unnecessary. Outputs include the imputed matrix and divergence history.

funzioneNowcast.m  
A versatile nowcasting framework integrating Sinkhorn-based imputation and classical EM-based approaches for handling missing data and dynamic factor estimation.

Key Components:  
- Data preprocessing with standardization and spline-based imputation  
- Dual-mode support: Sinkhorn-based or classical EM approaches  
- Iterative EM updates for system matrices estimation (`A`, `C`, `R`, `Q`)  
- Kalman filtering and smoothing for final state estimation  

Tracks log-likelihood to monitor convergence and outputs include estimated system matrices, log-likelihood, and reconstructed data.

funzioneNowcast_IS.m  
An extended nowcasting framework incorporating idiosyncratic dynamics into the model. This version expands the system matrices and integrates idiosyncratic noise for richer data modeling.

Additional Features:  
- Idiosyncratic dynamics with customizable lags (`s`)  
- Expanded system matrices to account for both factors and idiosyncratic components  
- EM algorithm modifications for joint factor and idiosyncratic estimation  
- Compatible with Sinkhorn-based and classical EM approaches  

Outputs include the smoothed state estimates, updated system matrices, and reconstructed data with idiosyncratic adjustments.

funzioneNowcastClassic.m 
A baseline implementation of the nowcasting framework using a classical EM approach without Sinkhorn imputation or idiosyncratic dynamics.

Core Features:  
- Simple dynamic factor model with EM parameter estimation  
- Missing value handling via spline-based imputation  
- Kalman filtering and smoothing for state estimation  

Designed as a benchmark for comparison against advanced nowcasting methods. Outputs the estimated system matrices, log-likelihood, and reconstructed data.

EMalgorithm.m, EMalgorithm_IS.m, EMalgorithm_MQ.m  
These functions implement the EM algorithm for estimating dynamic factor model parameters under varying settings:  
- `EMalgorithm.m`: Standard DFM with factors and lags.  
- `EMalgorithm_IS.m`: Extends to include idiosyncratic dynamics.  
- `EMalgorithm_MQ.m`: Adapts for mixed-frequency data (monthly and quarterly).  

Each version computes sufficient statistics using a Kalman smoother and updates system matrices iteratively.

initialize_EM.m, initialize_EM_Classic.m, initialize_EM_IS.m, initialize_EM_MQ.m  
These functions initialize system matrices and parameters for the EM algorithm:  
- `initialize_EM.m`: Standard initialization for DFM.  
- `initialize_EM_Classic.m`: Initialization for classical factor models.  
- `initialize_EM_IS.m`: Supports idiosyncratic dynamics.  
- `initialize_EM_MQ.m`: Prepares for mixed-frequency data models.  

Initialization relies on PCA and least-squares regression for observation and transition equations.

kalman_filter.m and Kalman_smoother.m 
Implement the Kalman filter and smoother for estimating states and variances in dynamic factor models. Essential for both the expectation step of the EM algorithm and final state estimation.

check_convergence.m 
Monitors the EM algorithm's convergence by evaluating the change in log-likelihood between iterations. Ensures early termination when convergence criteria are met.

remNaNs_spline.m  
Imputes missing values in the dataset using spline interpolation, aiding the initialization of the EM algorithm.

simulazioniBasicFunzione.m, simulazioniISFunzione.m, simulazioniNonParam.m, simulazioniBasicExpandingFunzione.m, simulazioniISBasicExpandingFunzione.m  
These functions implement simulation frameworks as callable functions for parallel processing. They replicate the main simulation scripts, enabling parallel execution with `parfor`.  
- Expanding functions support simulations with expanding windows.  
- NonParam functions evaluate non-parametric methods (KNN, regression trees, Sinkhorn).  
- IS functions include idiosyncratic dynamics in simulations.  

Remember to use `genpath` to generate the full path of subdirectories and `addpath` to include the necessary directories in your MATLAB environment for accessing all required functions and scripts.
