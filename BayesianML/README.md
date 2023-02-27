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
In the Bayesian approach, everything is a random variable, including the mean and variance. Our job is to find their probability distributions instead of point estimates. That is, we are more interested in finding a distribution instead of a number. That is, we want to find p(mean, variance | X).

By working with distributions, we can find intervals of confidence (not called confidence intervals in the Bayesian approach). That is, we can ask questions like "What is the probability that A is better than B?"

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

## Concepts Review

### Bayes Rule and Probability
[Notes](https://github.com/hangwl/DataScience/blob/master/Bayesian%20ML/Probability%20and%20Bayes%20Review.pdf)

### Definition of Machine Learning
In statistical learning, we build a model where the parameters of the model are learned from data. In MLE, we start by making a modeling assumption such as a Gaussian. We then find the parameters of the model using the data we collected. Linear regression and deep neural networks are examples of machine learning that use MLE. The broad definition of machine learning is similar, in that we aim to build a model with parameters learned and updated from data. 

In 'online learning', we use models that act in real-time. Data is ingested one sample at a time, and parameters are updated each time. Thus, the algorithm becomes smarter and smarter for each subsequent datapoint that collected.

## Traditional A/B Testing
Traditional A/B testing makes use of the Frequentist approach using point estimates to model a population's distribution.
We typically decide on how much data samples to collect beforehand, but prior to experimentation, it may be difficult to determine the power and effect size to do so.

Essentially, we are concerned with constructing confidence intervals to test our alternative hypothesis.
The p-value tells us whether the our alternative hypothesis is statistically significant, given a set level of significance (significance threshold).
It is the probability to reject the null when the null is true, or how likely that the data observed is to have occurred under the null hypothesis.
So, for example given a p-value of 0.03 < 5%, we should reject the null in favour of the alternative.

However, the definition of p-value is sometimes misleading.
For example if a p-value was instead 0.10 > 5%, we say that there is insufficient evidence to reject the null.
It is important to note that this does not imply that the null is true.

### Confidence Intervals
2 things affect our confidence in an estimate:
1. Variance - higher variance = less confident
2. Sample Size - larger number of samples = more confident

## Bayesian A/B Testing
Addressing the explore-exploit dilemma.

Suppose our goal is to find out which advertisement generates higher clicks, and then subsequently show more of that advertisement.
It is not practical to conduct a traditional Frequentist statistical AB test which predetermines the number of samples to be used.
Exploration is costly and any statistical experiment that is prematurely stopped will be invalidated.
So, the better approach would be to find an optimal balance between exploration and exploitation.

In the Bayesian Bandits (Thompson Sampling) approach, we can model and compare the distribution of true means.
To do this, we can make use of conjugate priors that take advantage of proportionality to find the distribution of the posterior.
That is, given our likelihood (bernoulli in the case of CTRs), we can search for its respective conjugate prior that has the same distribution.
By proportionality, the posterior will also have the same distribution.

Here is how our experiment runs in real-time:
When a user browses the website, a posterior sample is drawn from each bandit.
The bandit with the larger valued sample will be selected, and the respective advertisement will be shown to attempt to maximize our reward.
Thereafter, we log whether or not the user has clicked on the advertisement, and we update our posterior accordingly.