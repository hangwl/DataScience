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
			- [Supervised](#supervised)
				- [Linear Regression](#linear-regression)
				- [Logistic Regression](#logistic-regression)
				- [k-Nearest Neighbours](#k-nearest-neighbours)
				- [Support Vector Machines](#support-vector-machines)
				- [Decision Trees and Random Forests](#decision-trees-and-random-forests)
				- [AdaBoost/Gradient Boosting Machines](#adaboostgradient-boosting-machines)
				- [Neural Networks](#neural-networks)
			- [Unsupervised](#unsupervised)
				- [Clustering](#clustering)
				- [Visualization and Dimensionality Reduction](#visualization-and-dimensionality-reduction)
				- [Anomaly Detection](#anomaly-detection)
		- [Type of Learning](#type-of-learning)
			- [Batch Learning](#batch-learning)
			- [Online Learning](#online-learning)
			- [Transfer Learning](#transfer-learning)
			- [Active Learning](#active-learning)
			- [Ensembling](#ensembling)
		- [Underfitting](#underfitting)
		- [Overfitting](#overfitting)
			- [Regularization](#regularization)
		- [Hyperparameter Tuning](#hyperparameter-tuning)
			- [Learning Rate](#learning-rate)
			- [Number of Layers](#number-of-layers)
			- [Batch Size](#batch-size)
			- [Number of Trees](#number-of-trees)
			- [Number of Iterations](#number-of-iterations)
			- [Others](#others)
	- [Evaluation](#evaluation)
		- [Evaluation Metrics](#evaluation-metrics)
			- [Classification](#classification)
			- [Regression](#regression)
			- [Task-based Metric](#task-based-metric)
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
* Use [GitHub - scikit-learn-contrib/imbalanced-learn: A Python Package to Tackle the Curse of Imbalanced Datasets in Machine Learning](https://github.com/scikit-learn-contrib/imbalanced-learn)
	* [SMOTE](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/) (synthetic minority over-sampling technique) creates synthetic samples of the minor class which attempts to balance the dataset 
* Helpful paper to look at - [Learning from imbalanced data.](https://www.jeremyjordan.me/imbalanced-data/)

### Data Splitting
* Training set - 80% of dataset used to train model
* Validation set - 10% of dataset used to tune model hyperparameters
* Test set - final 10% to evaluate final performance of model

## Training
*3 steps: 1. Choose an Algorithm 2. Overfit the model 3. Reduce overfitting with regularization*
### Choosing an Algorithm

#### Supervised
##### Linear Regression
* [Linear regression - Wikipedia](https://en.wikipedia.org/wiki/Linear_regression)
##### Logistic Regression
* [Logistic regression - Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
##### k-Nearest Neighbours
* [k-nearest neighbors algorithm - Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
##### Support Vector Machines
* Read [Support Vector Machine — Simply Explained | by Lilly Chen | Towards Data Science](https://towardsdatascience.com/support-vector-machine-simply-explained-fee28eba5496)

##### Decision Trees and Random Forests
* [Random forest - Wikipedia](https://en.wikipedia.org/wiki/Random_forest)
* [How to visualize decision trees](https://explained.ai/decision-tree-viz/index.html)
* [1.10. Decision Trees — scikit-learn 1.2.1 documentation](https://scikit-learn.org/stable/modules/tree.html)

##### AdaBoost/Gradient Boosting Machines
* [A Gentle Introduction to the Gradient Boosting Algorithm for Machine Learning - MachineLearningMastery.com](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)
	* [XGBoost Documentation — xgboost 2.0.0-dev documentation](https://xgboost.readthedocs.io/en/latest/)
	* [CatBoost - open-source gradient boosting library](https://catboost.ai/)
	* [Welcome to LightGBM’s documentation! — LightGBM 3.3.5.99 documentation](https://lightgbm.readthedocs.io/en/latest/)
	
##### Neural Networks
* [Artificial neural network - Wikipedia](https://en.wikipedia.org/wiki/Artificial_neural_network)
	* [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
	* [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
	* [🤗 Transformers](https://huggingface.co/docs/transformers/index)

#### Unsupervised
##### Clustering
* [K-means clustering - Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)

##### Visualization and Dimensionality Reduction
* [Principal component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) 
	* Reducing data dimensions whilst attempting to preserve variance.
* [Autoencoders](https://www.jeremyjordan.me/autoencoders/) 
	* Learn a lower dimensional encoding of data. E.g. Compress an image of 100 pixels into 50 pixels representing (roughly) the same information as the 100 pixels.
* [t-distributed stochastic neighbor embedding (t-SNE)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
	* Good for visualizing high-dimensionality in a 2D or 3D space.
	
##### Anomaly Detection
* Use an autoencoder to reduce the dimensionality of the inputs of a system and then try to recreate those inputs within some threshold. If the recreations aren't able to match the threshold, there could be some sort of outlier.
* [One-Class Classification](https://machinelearningmastery.com/one-class-classification-algorithms/) 
	* Train model on only one-class. If anything lies outside of this class, it may be an anomaly. Algorithms for doing so include, one-class K-Means, one-class SVM, isolation forest, and local outlier factor.

### Type of Learning
#### Batch Learning
* Training of ML models in an (offline) batch manner.
* Models trained using batch learning are moved into production only at regular intervals based on the performance of models trained new data.
#### Online Learning
* Training happens in an incremental manner by continuously feeding data as it arrives or in a small group (mini-batches).
* Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives (as opposed to batch learning).
* Runs in production and learns continuously.
#### Transfer Learning
* ML method where a model developed for a task is reused as the starting point for a model on a second task.
* Popular approach in deep learning where pre-trained models are used as the starting point on computer vision and NLP tasks given the vast computational resources and time required to developed neural network models on these problems and from the huge jumps in skill that they provide on related problems.
	* Use [TensorFlow Hub](https://www.tensorflow.org/hub/) and [PyTorch Hub | PyTorch](https://pytorch.org/hub/) for broad model options
	* Use [🤗 Transformers](https://huggingface.co/docs/transformers/index) and [GitHub - Detectron2](https://github.com/facebookresearch/detectron2) for specific models
#### Active Learning
* [Active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) is a special case of machine learning in which a learning algorithm can interactively query a user (or some other information source) to label new data points with the desired outputs.
* E.g. [How Can Active Learning Help Train Autonomous Vehicles? | NVIDIA Blog](https://blogs.nvidia.com/blog/2020/01/16/what-is-active-learning/)

#### Ensembling
* [Ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning) uses multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.

### Underfitting
*Happens when our model doesn't perform as well as we like on the data. Can try training longer or with a more advanced model.*

### Overfitting
*Happens when our validation loss starts to increase, or when model performs far better on the training set than on the test set. Fix through regularization techniques.*

#### Regularization
* L1 (Lasso) and L2 (Ridge) Regularization
	* See [Differences between L1 and L2 as Loss Function and Regularization](http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/)
		* L1 regularization sets unneeded feature coefficients to 0 (performs feature selection on which features are most essential and which aren't, useful to improve model explanability).
		* L2 constrains model features (without setting to 0).
* Dropout Regularization
	* Randomly remove parts of the model to improve the rest of the model.
	* [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
* Early Stopping
	* Stop model from training before validation loss starts to increase too much or more generally, any other metric has stopped improving. Early stopping is usually implemented in the form of a model callback.
	* [tf.keras.callbacks.EarlyStopping  |  TensorFlow v2.11.0](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
* Data Augmentation
	* Manipulate dataset in artificial ways to make it 'harder to learn'. E.g. when dealing with images, randomly rotate, skew, flip and adjust the height of the images. This forces the model to have learn similar pattens across different styles of the same image.
		* [tf.keras.preprocessing.image.ImageDataGenerator  |  TensorFlow v2.11.0](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
		* [EDA - text augmentation with Python (boost performance on text classification tasks)](https://github.com/jasonwei20/eda_nlp)
		* [TextAttack - frame for adversarial attacks, data augmentation and model training in NLP](https://github.com/QData/TextAttack)
* Batch Normalization
	* Standardize inputs (zero mean and normalize) as well as adding two parameters (beta; epsilon) before they enter the next layer.
	* Often results in faster training speeds since the optimizer has less parameters to update. May replace dropout in some networks.

### Hyperparameter Tuning
* Very useful paper - [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)

#### Learning Rate
* Learning rate is often the most important hyperparameter, which controls how much to change the model in response to the estimated error each time the model weights are updated.
* High learning rate algorithms generally adapt to new data quickly, but the resulting weights may be suboptimal, and the training process may be unstable.
* Methods to optimize learning rate:
	* Train model for a few hundred iterations starting with a very low learning rate (e.g. 10e-6) and slowly increase it to a very large value (e.g. 1). Then, plot the loss versus the learning rate (using a log scale for learning rate). We should see a U-shaped curve, and the optimal learning rate is about 1-2 notches to the left of the bottom of the U-curve.
	* Learning Rate Scheduling (using Adam Optimizer - an extension to stochastic gradient descent).
	* [Cyclical Learning Rates](https://arxiv.org/abs/1506.01186) dynamically change the learning rate up and down between some threshold to potentially speed up training.

#### Number of Layers
* Number of layers to be used in deep learning networks.
#### Batch Size
* How many examples of data our model sees as once. Generally, we ought to use the largest batch size we can fight in our GPU memory. 
* If in doubt, use batch size 32. See [Revisiting Small Batch Training for Deep Neural Networks](https://arxiv.org/abs/1804.07612)
#### Number of Trees
* Number of trees used in decision tree algorithms.
#### Number of Iterations
* Instead of tuning model training iterations, use early-stopping instead.
#### Others
* Depending on the type of algorithm, they may be other hyperparameters used.
* Search 'algorithm_name' hyperparameter tuning.

## Evaluation

### Evaluation Metrics
#### Classification
* Accuracy
* Precision
* Recall
* f1
* Confusion Matrix
* Mean Average Precision (object detection)

#### Regression
* MSE (mean squared error)
* MAE (mean absolute error)
* R2 (r-squared)

#### Task-based Metric
* Depends on the specific problem.

### Feature Importance
* Which features contributed the most to the model? Should some be removed? Useful for model explainability. 

### Training/inference time/cost
* How long does a model take to train? Is this feasible?
* How long does inference take? Is it suitable for production?

### Model comparison
* How does the model compare to other models?
* What if I changed something in the data? [What-If Tool](https://pair-code.github.io/what-if-tool/)

### What does the model get wrong?
* Usually instances we don't have much observations for.

### Bias/Variance trade-off
* High bias results in underfitting and a lack of generalization to new samples, high variance results in overfitting due to model finding patterns in the data which is actually random noise.
* Optimal balance of bias and variance leads to model that is neither overfit nor underfit. [WTF is the Bias-Variance Tradeoff? (Infographic)](https://elitedatascience.com/bias-variance-tradeoff)

## Model Deployment
* Put model into production and see how it performs.
* Tools:
	* [Serving Models  |  TFX  |  TensorFlow](https://www.tensorflow.org/tfx/guide/serving)
	* [1. TorchServe — PyTorch/Serve master documentation](https://pytorch.org/serve/)
	* [Vertex AI  |  Google Cloud](https://cloud.google.com/vertex-ai) makes our model available as a REST API
	* [Machine Learning – Amazon Web Services](https://aws.amazon.com/sagemaker/)

## Retraining
* Machine learning is an iterative process where we must track our evaluation metrics, data and experiments, revisiting steps above as required.
* Model predictions tend to 'drift' naturally over time (due to environmental changes). Hence, the need for [Model Retraining - ML in Production](https://mlinproduction.com/model-retraining/).