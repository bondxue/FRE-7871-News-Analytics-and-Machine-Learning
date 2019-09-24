## NLP Question Preparation

--------------------------
1. What is NLP?

    Natural Language Processing or NLP is an automated way to understand or analyze the natural languages and extract required information from such data by applying machine learning Algorithms.
    
2. What is the significance of TF-IDF?

    **tf–idf** or **TFIDF** stands for **term frequency–inverse document frequency**. In information retrieval TFIDF is a numerical statistic that is intended to reflect how important a word is to a document in a collection or in the collection of a set.
    
3.  How can machines make meaning out of language

Popular NLP procedure is to use **stemming** and **lemmatization** methods along with the parts of speech tagging. The way humans use language varies with context and everything can’t be taken too literally.

**Stemming** approximates a word to its root i.e identifying the original word by removing the plurals or the verb forms. For example, ‘rides’ and ‘riding’ both denote ‘ride’. So, if a sentence contains more than one form of ride, then all those will be marked to be identified as the same word. Google used stemming back in 2003 for its search engine queries.

Whereas, **lemmatization** is performed to correctly identify the context in which a particular word is used. To do this, the sentences adjacent to the one under consideration are scanned too. In the above example, riding is the lemma of the word ride.

Removing stop words like a, an, the from a sentence can also enable the machine to get to the ground truth faster.

    
4. What does a NLP pipeline consist of?

Any typical NLP problem can be proceeded as follows:

Text gathering(web scraping or available datasets)

Text cleaning(stemming, lemmatization)

Feature generation(Bag of words)

Embedding and sentence representation(word2vec)

Training the model by leveraging neural nets or regression techniques

Model evaluation

Making adjustments to the model

Deployment of the model.

### What is cross validation? 

Cross-validation is a technique for evaluating ML models by training several ML models on subsets of the available input data and evaluating them on the complementary subset of the data. Use cross-validation to detect overfitting, ie, failing to generalize a pattern.

### What is Precision/ Recall?

Recall is also known as the true positive rate: the amount of positives your model claims compared to the actual number of positives there are throughout the data. Precision is also known as the positive predictive value, and it is a measure of the amount of accurate positives your model claims compared to the number of positives it actually claims. It can be easier to think of recall and precision in the context of a case where you’ve predicted that there were 10 apples and 5 oranges in a case of 10 apples. You’d have perfect recall (there are actually 10 apples, and you predicted there would be 10) but 66.7% precision because out of the 15 events you predicted, only 10 (the apples) are correct.


### What’s the trade-off between bias and variance?
Bias is error due to erroneous or overly simplistic assumptions in the learning algorithm you’re using. This can lead to the model underfitting your data, making it hard for it to have high predictive accuracy and for you to generalize your knowledge from the training set to the test set.

Variance is error due to too much complexity in the learning algorithm you’re using. This leads to the algorithm being highly sensitive to high degrees of variation in your training data, which can lead your model to overfit the data. You’ll be carrying too much noise from your training data for your model to be very useful for your test data.

The bias-variance decomposition essentially decomposes the learning error from any algorithm by adding the bias, the variance and a bit of irreducible error due to noise in the underlying dataset. Essentially, if you make the model more complex and add more variables, you’ll lose bias but gain some variance — in order to get the optimally reduced amount of error, you’ll have to tradeoff bias and variance. You don’t want either high bias or high variance in your model.

### How is KNN different from k-means clustering?

K-Nearest Neighbors is a supervised classification algorithm, while k-means clustering is an unsupervised clustering algorithm. While the mechanisms may seem similar at first, what this really means is that in order for K-Nearest Neighbors to work, you need labeled data you want to classify an unlabeled point into (thus the nearest neighbor part). K-means clustering requires only a set of unlabeled points and a threshold: the algorithm will take unlabeled points and gradually learn how to cluster them into groups by computing the mean of the distance between different points.

The critical difference here is that KNN needs labeled points and is thus supervised learning, while k-means doesn’t — and is thus unsupervised learning.

###  What is Bayes’ Theorem? How is it useful in a machine learning context?

Bayes’ Theorem gives you the posterior probability of an event given what is known as prior knowledge.

Mathematically, it’s expressed as the true positive rate of a condition sample divided by the sum of the false positive rate of the population and the true positive rate of a condition. Say you had a 60% chance of actually having the flu after a flu test, but out of people who had the flu, the test will be false 50% of the time, and the overall population only has a 5% chance of having the flu. Would you actually have a 60% chance of having the flu after having a positive test?

Bayes’ Theorem says no. It says that you have a (.6 * 0.05) (True Positive Rate of a Condition Sample) / (.6\*0.05)(True Positive Rate of a Condition Sample) + (.5\*0.95) (False Positive Rate of a Population)  = 0.0594 or 5.94% chance of getting a flu.
