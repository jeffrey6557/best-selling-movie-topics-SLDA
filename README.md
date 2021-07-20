# Best-selling Movie Topic Discovery using SLDA
### Project motivation
Designed a method to help entertainment firms source profitable movie ideas; implemented supervised and semi-supervised LDA on IMDb movie reviews to find best-selling movie topics and predict box office performance based on a learned movie topic.

This project requires the Large Movie Review Dataset which you can download from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz. 

To run the analysis step by step:
1. Download the dataset from above and unzip the dataset at directory the directory /SLDA/data/. 
2. run these scripts in sequence:
 - download_sales.py to scrap movie sales data from Box Office Mojo and save at /SLDA/data/sales_data/
 - preprocess_corpus.py to label and process the movie review data (and optionally analysis.py which contains code for exploratory analysis)
 - run_experiments.py to run experiments on various settings (models will be cached at SLDA/models/
 - analysis.py to visualize the prediction results

There are two questions of interest here:
1. The theoretical contribution of the supervised LDA (SLDA) is that when training a LDA model, adding label information about the documents help build topics that 'shift' towards the dimension of labels, and therefore the "weight" vector SLDA produces are more predictive of the labels. In this case, I want to see whether the topics SLDA build can predict box office performance of a movie title given its documents.
2. Once we have built such a SLDA model, we can also look at whether the topic with the highest weights are indeed best-selling by human inspection. Of course, this would be somewhat subjective because we are the judges! Empirically, however, this is somewhat mitigated by the more refined topics selected by hyperparameter tuning that point to very specific titles that we can look up its ticket sales directly. 

Now let's start with the high-level results:

### How well can the SLDA model predict box-office performance out of samples?

The bar plot below shows the percentage difference of prediction efficacy metrics (MAE and MSE) in various data modes (e.g. train vs validation). The positive bars of MAE in test data both indicate that SLDA outperforms LDA out of samples, while the negative bar MSE indicates that it is underperforming. The difference in result suggests SLDA performs worse in face of outlier box-office performance, which could be prevalent.

<img width="709" alt="Screen Shot 2021-04-11 at 11 55 09 PM" src="https://user-images.githubusercontent.com/9246300/114338584-5c517380-9b21-11eb-8049-d4bf4cd65e4d.png">

<img width="713" alt="Screen Shot 2021-04-12 at 12 04 26 AM" src="https://user-images.githubusercontent.com/9246300/114339212-a8e97e80-9b22-11eb-8558-2c4f2b4afb3c.png">


### What are the best-selling topics? Are they your favorites?
  
Best-selling movie topics found by SLDA trained with 50 topics out of 22k movie titles, with the top 10 words listed for each topic:

The best-selling topic ['benjamin', 'pitt', 'fincher', 'brad', 'freeman', 'bride', 'button', 'hudson', 'wed', 'walken'] seems to loads on Brad Pitt's classic The Curious Case of Benjamin Button (directed by David Fincher) with $335 million sales.  

The runner-up topic ['alien', 'earth', 'robot', 'space', 'sci-fi', 'shark', 'human', 'ship', 'scienc', 'planet'] is clearly referring to the Aliens series and also the Jaws series, both of which are extremely successful, with a whopping $240 million and $470 million sales in just one of their sequels, respectively.

It is also interesting to note that these are not among the single biggest hits - Avatar with a whopping 2.8 billion sales is the champion and Titatnic with a 2.2 billion is the runner-up! So the model is not picking the obvious answer and there could be some degree of generalization here. More on this later.

### Modeling discussions: the nitty-gritty 


Looking at the generalization gap plot, it points to some degree of overfitting: as the number of topics increases, the generalization gap between training and validation sets increases while the training and validation performance improve together. For this simple exercise, the hyper-parameter tuning process is perhaps too simplistic which focuses on the number of topics, while other hyper-parammeters such as number of iterations, and minimum and maximum word frequency filters are left untuned, possibly resulting in a too diverse set of vocabulary (11k words) and thus too movie-specific topics that are hard to generalize well from the training set to the test set. For example, if the alien movies are the best-selling topic in the training set, but the test set may not even have any movies related to aliens. Then the model is not allowed to learn such a specific topic, although from a movie producer's perspective, this is a perfectly acceptable answer. Instead, the model will be forced to learn very high-level topic such as comedy or action, which are overly broad genre and is certainly less useful for our purposes. 


## Reference
 
[1] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

[2] David M. Blei and Jon McAuliffe. Supervised topic models. In NIPS, 2007.

[3] https://github.com/bab2min/tomotopy
