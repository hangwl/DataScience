Credits: [A/B Testing (lazyprogrammer)](https://github.com/lazyprogrammer/machine_learning_examples/tree/master/ab_testing)

# A/B Testing
A/B Testing is the most readily applicable of all ML ideas.

## Real-World Examples of A/B Testing
Generally, A/B testing is all about comparing 2 or more items (something any business is interested in). It includes application of statistical hypothesis testing or "two-sample hypothesis testing" as used in the field of statistics. A/B testing is a way to compare multiple versions of a single variable, for example by testing a subject's response to variant A against variant B, and determining which of the variants is more effective.

### Medicine
A pharmaceutical company discovered a new drug and wants to find out whether it works. Let assume that the drug is for reducing blood pressure, then we want to know if the drug reduces blood pressure to a significant degree. To find the answer, we will need to run an experiment with 2 groups, the treatment(drug) and control group(placebo), and then measure their blood pressure. (Note that realistically, we would want to measure the difference between before and after taking the drug.) How can we give a definitive answer about whether the drug is working?

### Making a Website
For websites that run businesses, they typically keep track of some form performance metric, e.g. conversion rate. That is, out of of 100 visitors, 1 might buy something implies a 1% conversion rate. Businesses would surely want to maximize their conversion rate, and some might choose to hire a copywriter to write some content for their sales page. The one that gives the highest conversion rate is the one we want to pick.

### Local Business Flyers
A designer makes several flyers to hand out in the neighborhood. The best flyer is the one the returns the highest number of callbacks.

### Cloud Storage
A cloud storage service wants to buy hard drives and is deciding between Seagate and Western Digital. A/B testing helps us choose the hard drive with the lower failure rate to minimize expenditure costs.

## Frequentist Approach vs Bayesian Approach
The frequentist approach assumes data are random sample and parameters are fixed while the Bayesian approach assumes both data and parameters are random.

Suppose we want to model the height of students in a class as a Gaussian (normal distribution), we need to find the mean and variance.

### The Frequentist Approach
In the Frequentist approach* we first collect data of every student in the class. Then, using MLE (maximum likelihood estimation), we find the mean and variance. To do this, we first calculate the likelihood function and then maximize the likelihood with respect to the gaussian mean and variance.

### The Bayesian Approach
In the Bayesian approach, everything is a random variable, including the mean and variance. Our job is to find their probability distributions. That is, we are more interested in finding a distribution instead of a number. That is, we want to find p(mean, variance | X).

In the statistics route, a lot of our attention will be placed sampling methods such as importance sampling, MCMC, Gibbs, etc. The idea behind sampling methods is to do numerical approximation of an integral. Most people who talk about Monte Carlo talk about it as a technique of integration rather than a technique of statistics.

In Bayesian Machine Learning Models, we are not interested in finding the vector w (as in y = w * x in linear regression). Rather we are more interested in the distribution of the 'random' vector w. That is, we want to find p(w | X, Y), where X and Y is our training data. The same concept is applied in many classic machine learning models:
* Logistic Regression
* Neural Networks
* Gaussian Mixture Models (a more powerful k-means clustering)
* PCA (principal components analysis)
* Matrix Factorization (for recommender systems)

Bayesian Networks can be used to model dependencies explicitly (as opposed to neural networks where we let the neural network decide on what the weights should be), that is we can model for causation based on our understanding of the system. 
* Latent Dirichlet Allocation (LDA) is a famous example of Bayes nets.
	* LDA is an algorithm used for topic modeling
	* It is a specific example of a Bayes net, but does not have its roots in a non-Bayesian model such as a regular Logistic Regression
	* Also a totally different subset of ML (NLP)

## Bayes Rule and Probability Review
[Notes](https://github.com/hangwl/DataScience/blob/master/Bayesian%20ML/Probability%20and%20Bayes%20Review.pdf)
## Traditional A/B Testing

## Bayesian A/B Testing

## Bayesian A/B Testing Extension