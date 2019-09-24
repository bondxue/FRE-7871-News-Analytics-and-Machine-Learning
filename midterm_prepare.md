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

### Why is “Naive” Bayes naive?

Despite its practical applications, especially in text mining, Naive Bayes is considered “Naive” because it makes an assumption that is virtually impossible to see in real-life data: the conditional probability is calculated as the pure product of the individual probabilities of components. This implies the absolute independence of features — a condition probably never met in real life.


### Explain the difference between L1 and L2 regularization.

L2 regularization tends to spread error among all the terms, while L1 is more binary/sparse, with many variables either being assigned a 1 or 0 in weighting. L1 corresponds to setting a Laplacean prior on the terms, while L2 corresponds to a Gaussian prior.

### What’s the difference between a generative and discriminative model?

A generative model will learn categories of data while a discriminative model will simply learn the distinction between different categories of data. Discriminative models will generally outperform generative models on classification tasks.

### What cross-validation technique would you use on a time series dataset?

Instead of using standard k-folds cross-validation, you have to pay attention to the fact that a time series is not randomly distributed data — it is inherently ordered by chronological order. If a pattern emerges in later time periods for example, your model may still pick up on it even if that effect doesn’t hold in earlier years!

You’ll want to do something like forward chaining where you’ll be able to model on past data then look at forward-facing data.

+ fold 1 : training [1], test [2]
+ fold 2 : training [1 2], test [3]
+ fold 3 : training [1 2 3], test [4]
+ fold 4 : training [1 2 3 4], test [5]
+ fold 5 : training [1 2 3 4 5], test [6]

### How is a decision tree pruned?

Pruning is what happens in decision trees when branches that have weak predictive power are removed in order to reduce the complexity of the model and increase the predictive accuracy of a decision tree model. Pruning can happen bottom-up and top-down, with approaches such as reduced error pruning and cost complexity pruning.

Reduced error pruning is perhaps the simplest version: replace each node. If it doesn’t decrease predictive accuracy, keep it pruned. While simple, this heuristic actually comes pretty close to an approach that would optimize for maximum accuracy.

### What’s the F1 score? How would you use it?

The F1 score is a measure of a model’s performance. It is a weighted average of the precision and recall of a model, with results tending to 1 being the best, and those tending to 0 being the worst. You would use it in classification tests where true negatives don’t matter much.

### Name an example where ensemble techniques might be useful.

Ensemble techniques use a combination of learning algorithms to optimize better predictive performance. They typically reduce overfitting in models and make the model more robust (unlikely to be influenced by small changes in the training data). 

You could list some examples of ensemble methods, from bagging to boosting to a “bucket of models” method and demonstrate how they could increase predictive power.

### How do you ensure you’re not overfitting with a model?

This is a simple restatement of a fundamental problem in machine learning: the possibility of overfitting training data and carrying the noise of that data through to the test set, thereby providing inaccurate generalizations.

There are three main methods to avoid overfitting:

1- Keep the model simpler: reduce variance by taking into account fewer variables and parameters, thereby removing some of the noise in the training data.

2- Use cross-validation techniques such as k-folds cross-validation.

3- Use regularization techniques such as LASSO that penalize certain model parameters if they’re likely to cause overfitting.

### What’s the “kernel trick” and how is it useful?

The Kernel trick involves kernel functions that can enable in higher-dimension spaces without explicitly calculating the coordinates of points within that dimension: instead, kernel functions compute the inner products between the images of all pairs of data in a feature space. This allows them the very useful attribute of calculating the coordinates of higher dimensions while being computationally cheaper than the explicit calculation of said coordinates. Many algorithms can be expressed in terms of inner products. Using the kernel trick enables us effectively run algorithms in a high-dimensional space with lower-dimensional data.

### What’s selection bias? What other types of biases could you encounter during sampling?

When you’re dealing with a non-random sample, selection bias will occur due to flaws in the selection process. This happens when a subset of the data is consistently excluded because of a particular attribute. This exclusion will distort results and influence the statistical significance of the test.

Other types of biases include survivorship bias and undercoverage bias. It’s important to always consider and reduce such biases because you’ll want your smart algorithms to make accurate predictions based on the data.

### What’s a random forest? Could you explain its role in AI?

A random forest is a data construct that’s applied to ML projects to develop a large number of random decision trees while analyzing variables.

These algorithms can be leveraged to improve the way technologies analyze complex data sets. The basic premise here is that multiple weak learners can be combined to build one strong learner.

This is an excellent tool for AI and ML projects because it can work with large labeled and unlabeled data sets with a large number of attributes. It can also maintain accuracy when some data is missing. As it can model the importance of attributes, it can be used for dimensionality reduction.


### What’s regularization?

When you have underfitting or overfitting issues in a statistical model, you can use the regularization technique to resolve it. Regularization techniques like LASSO help penalize some model parameters if they are likely to lead to overfitting.

If the interviewer follows up with a question about other methods that can be used to avoid overfitting, you can mention cross-validation techniques such as k-folds cross-validation.

Another approach is to keep the model simple by taking into account fewer variables and parameters. Doing this helps remove some of the noise in the training data.

### What is NLP(natural language processing) ?

Natural language processing is a subfield of computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human languages, in particular how to program computers to process and analyze large amounts of natural language data


### What is tokenization ?

Splitting the sentence into words

### What is stemming ? 

Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes.

### What is lemmatizing ? 

Lemmatizing is also same like stemming but the difference is lemmantizing words known with dictionary.

### What is Normalization ? 

Converting different range of values to same scale from 0 to 1.

### What are nlp libraries and tools ? 

CoreNLP from Stanford group.

NLTK, the most widely-mentioned NLP library for Python.

TextBlob, a user-friendly and intuitive NLTK interface.

Gensim, a library for document similarity analysis.

SpaCy, an industrial-strength NLP library built for performance.

### What are stop words ? 

a, the , an etc like repeated words in text, that doesn’t give any additional value to context. we can filter those words by using nltk library standard function.

### What is Corpus ? 

It’s a collection of text documents.

### What is N- Gram, Unigram, Bigram  and Trigram? 

it’s about word analysis, unigram means single word, bigram means double words and trigram means tripple word.

### What is word embedding ?
Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing where words or phrases from the vocabulary are mapped to vectors of real numbers

### What are word embedding libraries ? 

+ Word2vec
+ Glove
+ Fasttext
+ genism

### What is word2vec ? 

Word2vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words

### What is Topic Modeling ? When we will do it ?

Topic modeling is a type of statistical modeling for discovering the abstract “topics” that occur in a collection of documents. Latent Dirichlet Allocation (LDA) is an example of topic model and is used to classify text in a document to a particular topic

### What is text generation ? When we will do it ?

Generate new text from understanding old data.