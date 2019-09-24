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