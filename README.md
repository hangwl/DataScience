# Machine Learning Process

- [Machine Learning Process](#machine-learning-process)
	- [Data Collection](#data-collection)
		- [Questions to ask](#questions-to-ask)
	- [Types of Data](#types-of-data)
		- [Structured Data](#structured-data)
		- [Unstructured Data](#unstructured-data)
	- [Data Preparation](#data-preparation)
		- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
		- [Data Preprocessing](#data-preprocessing)
			- [Feature imputation](#feature-imputation)
			- [Feature Encoding](#feature-encoding)
			- [Feature Scaling or Standardization](#feature-scaling-or-standardization)
			- [Feature Engineering](#feature-engineering)
			- [Feature Selection](#feature-selection)
			- [Dealing with Imbalances](#dealing-with-imbalances)
		- [Data Splitting](#data-splitting)
	- [Training](#training)
		- [Choosing an Algorithm](#choosing-an-algorithm)
		- [Type of Learning](#type-of-learning)
		- [Underfitting](#underfitting)
		- [Overfitting](#overfitting)
		- [Hyperparameter Tuning](#hyperparameter-tuning)
	- [Evaluation](#evaluation)
		- [Evaluation Metrics](#evaluation-metrics)
		- [Feature Importance](#feature-importance)
		- [Training/inference time/cost](#traininginference-timecost)
		- [Model comparison](#model-comparison)
		- [What does the model get wrong?](#what-does-the-model-get-wrong)
		- [Bias/Variance trade-off](#biasvariance-trade-off)
	- [Model Deployment](#model-deployment)
	- [Retraining](#retraining)

## Data Collection

### Questions to ask
* What kind of problem are we trying to solve?
* What data sources already exist?
* What privacy concerns are there?
* Is the data public?
* Where should we store the data?

## Types of Data

### Structured Data
* Nominal/Categorical (e.g. Gender)
* Numerical (e.g. Weight)
* Ordinal (e.g. Satisfaction)
* Time Series (e.g. Average Daily Temperature)

### Unstructured Data
* Data with no rigid structure (e.g. images, video, natural language text, speech)

## Data Preparation

### Exploratory Data Analysis (EDA)
* Identify feature (input) variables and target (output) variables
* Identify missing values
* Identify outliers
* Consult a domain expert to gain insight into dataset

### Data Preprocessing

#### Feature imputation
* Simple imputation: Fill using mean, median of column
* Multiple imputation: Model for other missing values
* KNN imputation: Fill data with a value from another similar example
* Random imputation, last observation carried forward (for time series), moving window, most frequent, ...

#### Feature Encoding
* OneHotEncoding: Encode unique categorical values into dummy binary dummy variables
* LabelEncoder: Turn labels into distinct numerical values
* Embedding Encoding

#### Feature Scaling or Standardization
* Feature scaling (normalization) scales data points such that they lie between 0 and 1 by subtracting the min. value and dividing by (max - min)
* Feature standardization standardizes all values so that they have a mean of 0 and unit variance by subtracting the mean and dividing by the std. deviation of that particular feature. Compared to normalization, standardization is more robust to outliers.

#### Feature Engineering
*Feature engineering is essentially transforming data into (potentially) more meaningful representations by adding domain knowledge*
* Discretization (turning larger groups into smaller groups - e.g. binning features, combining features)
* Crossing features and interaction features (e.g. difference between 2 features)
* Indicator features (e.g. age < X)

#### Feature Selection
* Dimensionality reduction
	* Principal Component Analysis (PCA) uses linear algebra to reduce the number of dimensions (features) in the model
* Feature importance (post modelling)
	* Fit model, and remove the least important features from the model
* Wrapper methods
	* Genetic algorithms and recursive feature elimination involve creating large subsets of feature options and then removing the ones which don't matter. These have the potential to find a great set of features but can also require a large amount of computation time. 
		* See <[TPOT](http://epistasislab.github.io/tpot/)> - *TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.*

#### Dealing with Imbalances
*e.g. data contains 10,000 examples of one class but only 100 examples of another*
* Collect more data (if possible)
* Use <[GitHub - scikit-learn-contrib/imbalanced-learn: A Python Package to Tackle the Curse of Imbalanced Datasets in Machine Learning](https://github.com/scikit-learn-contrib/imbalanced-learn)>
	* SMOTE (synthetic minority over-sampling technique) creates synthetic samples of the minor class which attempts to balance the dataset
* Read <[Learning from imbalanced data.](https://www.jeremyjordan.me/imbalanced-data/)>

### Data Splitting
* Training set - 80% of dataset used to train model
* Validation set - 10% of dataset used to tune model hyperparameters
* Test set - final 10% to evaluate final performance of model

## Training
*3 steps: 1. Choose an Algorithm 2. Overfit the model 3. Reduce overfitting with regularization*
### Choosing an Algorithm
### Type of Learning
### Underfitting
### Overfitting
### Hyperparameter Tuning

## Evaluation
### Evaluation Metrics
### Feature Importance
### Training/inference time/cost
### Model comparison
### What does the model get wrong?
### Bias/Variance trade-off
* High bias results in underfitting and a lack of generalization to new samples, high variance results in overfitting due to model finding patterns in the data which is actually random noise.


## Model Deployment
* Put model into production and see how it performs.
* Tools:
	* TensorFlow Serving (part of TFX, TensorFlow Extended)
	* PyTorch Serving (TorchServe)
	* Google AI Platform: makes our model available as a REST api
	* Sagemaker
* MLOps

## Retraining
* Machine learning is an iterative process where we must track our evaluation metrics, data and experiments, revisiting steps above as required.
* Model predictions tend to 'drift' naturally over time (due to environmental changes). Hence, the need for model retraining.