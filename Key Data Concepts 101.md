# Data
1. Data - any information that can represented in the form of numbers that can be understood by computers
2. Structured Data - data that can be structured in a 2-dimensional table
3. CSV - common way of storing structured data
4. Semi-structured Data - data that is stored in a tagged hierarchical format that doesn't easily fit into a tabular or structured format
5. JSON - common way of storing semi-structured data
6. Unstructured Data - data that cannot be easily represented in a structured or semi-structured format
# Data Engineering
8. Data Engineer - conductor that ensures that data from many different sources comes together gracefully in a symphony that is clean, error free and computationally efficient
9. Database - computer systems that corporations use to stored organized data in all formats
10. Relational Database - the most popular form of database which is controlled by a relational database management system, which only stores structured data and separates its data into tables which are related to each other through the columns
11. ACID Transactions - one of the most useful aspects of relational databases and the reason why they're the most popular database format in the world due to being atomic, consistent,  isolated, and durable
	* atomic - each transaction read and write is treated as a separate unit, i.e. either the entire transaction is executed or it isn't, which protects against data loss and corruption
	* consistent - transactions are made in predefined a predictable ways i.e. if something in the data is corrupted, unintended errors in our table won't occur
	* isolated - multiple users reading and writing simultaneously will have their transactions isolated from one another ensuring concurrent transactions can happen without interference
	* durable - even if the system fails, transactions will be saved
12. SQL - relational database querying language
13. Filter - method to limit the amount of data to be extracted into a table (e.g. select X from DATA_SOURCE where X = ?)
14. Aggregate - taking a certain number of rows and combining the data in such a way that we will end up with less data than before (e.g. sum or average)
15. Union - method to vertically combine (or stack) data from multiple tables
16. Join - method to combine tables horizontally by matching values in common columns from multiple tables
17. NoSQL Database - usually not ACID compliant but can scale to handle massive amounts of traffic better than many SQL databases can e.g. MongoDB
18. On-Prem - self-hosted servers
19. Cloud Services - group of services managed by a large company who takes care of everything from buying and powering the servers to managing physical security to adding and subtracting server power based on how much traffic we have (e.g. AWS, Azure, GCP)
20. Snowflake - Cloud computing-based data cloud company that offers cloud-based data storage and analytics services (data-as-a-service) built on top of AWS
21. Data Warehouse - database that companies store organize large amounts of data away from their production systems or systems that actually run business processes so that it can be analyzed by data professionals
22. Big Data - term describing hard-to-manage data: 1. high volume 2. high velocity 3. high variety
23. Vertical Scaling - increasing the storage and processing power of our system (works relatively well at the individual level, but does not scale very well for large operations)
24. Horizontal Scaling - buying more commodity hardware and writing software that distributes the files and computational needs across multiple computers
25. Object Storage - a form of storage that allows you to dump a file into a sectioned off part of a disk and then gives that file a key that we can use to retrieve the data later, allowing us to create massive data lakes that can store vast amounts of data relatively cheaply to deal with the volume of data required in the modern world
26. Batch Data Processing - periodical processing of data, which requires simpler data infrastructure to set up
27. Streaming Data - in real time processing of data through a streaming data pipeline and then process it, which also allows for real-time analytics
28. Apache Kafka - a high-throughput messaging system that takes in all the data from input systems and coordinate their transportation to output systems
29. Apache Hadoop - big data distributed computing framework that can horizontal scale the computational power available to us and intelligently distributed the load across these systems
30. Apache Spark - designed to counteract the deficiencies of Hadoop, intended to be faster as it would use RAM to store intermediate operations instead of disk space (can be more expensive to run)
31. Apache Software Foundation - opensource community of volunteers who manage a lot of the software tools that undergird modern technological infrastructure
32. Open Source Software - software that can be freely used or modified for any purpose including commercial
# Data Science
33. Data Scientist - data scientists often focus on the complicated problems a company might have that can be solved with deep and computationally or mathematically intensive data, often times working on a very complicated problem where data is not easily available for over the course of multiple weeks or months and use tools from coding to mathematics to try and estimate and answer for a previously unanswerable question
34. Python - general purpose programming language that is valued  for its clean syntax, robust selection of add-ons (libraries) and its open source nature
35. R - another popular programming language for data scientists, popular amongst people with very strong academic backgrounds
36. Version Control - system that allows people who write code to have multiple versions of it which can be tested automatically for bugs prior to launching it, helping large teams work together in an error-free and integrated manner
37. Jupyter Notebook / R Markdown - format that allows the iterative running of code blocks in a single file
38. pip / CRAN - package manager to install external libraries
39. Numpy - popular package with pre-compiled C code passed through a Python wrapper that allows us to perform various array operations
40. DataFrame - programming equivalent of tabular data 
41. Pandas - library that adds dataframe functionality to python is pandas (short for PanelData), that allows us to import, read, clean, analyze, and export data using simple commands
42. SciPy - library that has many features useful for scientific computing, useful to solve optimization problems, linear algebra, integration, and interpolation
43. Tidyverse - R equivalent useful packages conveniently combined into a single library
44. Scikit Learn - standard library for various machine learning algorithms
45. Machine Learning - field dedicated to programming machines to teach themselves how to become better at a task after learning with some data
46. CRISP-DM - cross industry standard process for data mining, which holds up well  as a framework to tackle many data science problems
47. EDA - exploratory data analysis means taking statistics of data and check if there are any missing values and overall measuring the data quantitatively and qualitatively to fully understand it and if it will help solve our business question 
48. Data Preprocessing - after an agreement on a realistic business goal and its associated data has been reached, this is where we clean the data
49. Instance - each row of our tabular data refers to an instance (or observation) of our data
50. Feature - each column of our tabular data refers to a feature of our data
51. Imputation - data preprocessing technique where we try and estimate the value of missing values in our data
52. Encoding - step to map categorical data to numerical data
53. Feature Engineering - can be used to manipulate or combine features to try create better predictors for our data
54. Train-Test Split - important step in a data preprocessing pipeline, where we split our data into at least two sections (training and testing data set) so that we can determine the actual predictive power of our algorithm
55. Train - modeling process where we iterate over our training data to hopefully learn something through each iteration
56. Fit - the process of fitting the algorithm to our data
57. GPUs - graphics processing units capable of processing mathematical operations through large iterations of data
58. TPUs - tensor processing units are specialized to be even better at performing mathematical operations for certain types of ML algorithms than GPUs
59. Hyperparameter - settings that control the way an algorithm is trained, useful to fine-tune and improve the performance of a model
60. Dimensionality Reduction - method which makes use of algorithms to find dense patterns in our data that can be grouped together and remove features or combine features such that we can reduce the overall number of features we're working with, thereby reducing the dimensions to our data
61. Model Evaluation - final step of the preprocessing stage where we evaluate our model against the original business requirements to check whether we have to adjust our business goals or if we don't have the data required to solve the project
62. Model Deployment - releasing our model into the real world to solve whatever issue it was created to solve
63. Supervised Algorithms - takes a known set of input features (predictors that we might have) to train on a target variable of some kind
64. Target Variable - variable that we would like to predict
65. Regression - is used if our target variable is numeric
66. Classification - is used if our target variable is categorical
67. Binary Classification - predicting whether an object is one thing or another
68. Logistic Regression - a common type of binary classification algorithm that uses a logit function to model the target variable
69. Support Vector Machines - one of the most common algorithms in machine learning and works by trying to find the biggest gap between various groups inside a data set
70. Multi-label Classification - another type of classification where we can classify each instance with multiple labels (e.g. predicting both the make and model of a car bought by individuals based on their buying habits)
71. Unsupervised Algorithms - do not try to match our data to a specific target variable, but instead try to find general patterns in the data, useful for clustering data
72. Clustering - unsupervised technique where our algorithm will try to cluster observations into groups based on how related instances are to another
73. K-NN - k's nearest neighbors is a clustering technique which works by defining what k should be and the algorithm will create k centroids in our data, and group each nearest Euclidean data point to the respective centroid
74. Reinforcement Learning - is a machine learning algorithm that tries to teach by setting some desired outcome and rewarding the algorithm when the outcome is reached and punishing it if the outcome isn't reached
75. Decision Tree - a versatile yet common algorithm that tries to create a tree of decision nodes to classify values into different categories using a metric such as information gain to determine how to split each decision node into two different routes
	* relatively simple to explain, quick to train and yet fairly accurate
77. Ensemble Methods - most effective algorithms these days that try to combine the predictions of multiple algorithms to create one super algorithm
78. Random Forest - one of the simplest but most popular ensemble algorithm which works by training multiple decision trees to try and predict an outcome
	* Similar benefits to decision trees in that they're easy to implement pretty good predictors and easy to explain
79. Neural Network - series of algorithms that mimics the human brain, by sending inputs through multiple layers of nodes to the output layer, backpropagating to correct weights and biases until it reaches the answer
80. Backpropagation - algorithm used for training feedforward artificial neural networks by computing the gradient of the loss function with respect to the weights of the network for a single input-output example, and using it to update weights such that they minimize loss (e.g. stochastic gradient descent)
81. TensorFlow/PyTorch - libraries used to implement neural network models
82. Boosting - is an ensemble meta-algorithm for primarily reducing bias, and also variance in supervised learning, and a family of machine learning algorithms that convert weak learns to strong ones
83. Databricks - helps train machine learning algorithms by using cloud GPUs and big data frameworks to handle massive amounts of data without having to buy any expensive hardware
84. Model Drift - naturally happens when new data is introduced into a system or when conditions change, causing the model to become worse at predicting things over time
# Statistics
85. Descriptive Statistics - involve describing features of a data set by generating statistical summaries of the data samples
86. Inferential Statistics - is the process of using data analysis to infer properties of an underlying distribution of probability of a population
87. Bayesian Statistics - field of statistics that concerns itself with the likelihood of an event happening given other events have happened
88. Data Distribution - describes how (usually numerical) data is distributed
89. Selection Bias - type of bias introduced when individuals or instances selected for groups are not truly randomized which can lead to poor accuracy in our algorithms
90. Bootstrapping - a method to address such selection bias, and also try and make our algorithms work when we do not have much data to work with, which works by selecting a random sample, and replaces all items in the sample, and then sample again
91. Hypothesis Testing - a way to test the results of a survey or experiment to see if we have meaningful results by testing whether our results are valid by figuring out the odds that our results have happened by chance (if it happens by chance, then the experiment will unlikely to be repeatable so it has little use)
92. Data Analyst - generally works with business leaders to understand their needs in a process called requirements gathering, check if the data is available, analyze, and present it to said business leaders (to drive some form of change in an organization or group of people through data storytelling)
93. Data Storytelling - telling a story to communicate the message using data is generally more convincing to invoke change
	1. narrate change over time
	2. start big and drill down - great way to give context to smaller data points
	3. start small and zoom out - taking a individual example to anchor the audience and then zoom out to show how the problem affects a much large population
	4. highlight contrasts - great way of showing how problem areas tend to cluster around one another
	5. explore the intersection - can be how different phenomena develop in response to stimuli in their environment
	6. dissect the factors - we often want to know what the constituent parts an observed phenomena are
	7. profile the outliers - sometimes the outliers are what we're truly interested in (e.g. fraud detection)
94. Business Intelligence - concerned with using generally tabular data and transforming and visualizing it in order to explain business performance (more about communicating the data says in formats that help leaders understand how the business is performing)
95. BI Tools - graphical user interfaces that business professionals use to connect to different sources of data, making visualizing them easy
96. Matplotlib / Seaborn / ggplot2 - libraries used to visualize data in Python and R
97. Scatter Plots - commonly used to visual a relationship between two or more variables 
98. Bar Charts - staple of data visualization familiar to most people
99. Pie Charts - pizza charts sliced by volumes of each data category
100. Line Graphs - excellent way to show change over time
101. Time Series Data - very common data type used to graph line graphs, i.e. any data with a date and time in one column and generally a numeric value another column
102. Tree Maps - fun graph type that can highlight the biggest categories in a complex system (often used to map out the sectors of a nation's economy)
103. Histogram - special bar chart that groups values in a single group into bins that are represented by columns used to illustrate how data is distributed
104. Cloropleth Map - great way to illustrate the differences in geographic areas used a lot to illustrate the difference in populations
105. Radar Charts - great way of comparing multiple quantitative variables one another and highlighting the outliers that might be there

Credits: [Shashank Kalanithi](https://www.youtube.com/@ShashankData)