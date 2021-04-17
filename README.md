# Best-selling Movie Topic Discovery using SLDA
Designed a method to help entertainment firms source profitable movie ideas; implemented supervised and semi-supervised LDA on IMDb movie reviews to find best-selling movie topics and predict box office performance based on a learned movie topic.

This project requires the Large Movie Review Dataset which you can download from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz. 

To run the analysis step by step:
1. Download the dataset from above and unzip the dataset at the directory '/SLDA'. 
2. run these scripts in sequence:
 - download_sales.py to scrap movie sales data from Box Office Mojo
 - preprocess_corpus.py to label and process the movie review data
 - run_experiments.py to run experiments on various settings
 - analysis.py to do exploratory analysis and visualize the prediction results

There are two questions of interest here:
1. The theoretical contribution of the supervised LDA (SLDA) is that when training a LDA model, adding label information about the documents help build topics that 'shift' towards the dimension of labels, and therefore the "weight" vector SLDA produces are more predictive of the labels. In this case, I want to see whether the topics SLDA build can predict box office performance of a movie title given its documents.
2. Once we have built such a SLDA model, we can also look at whether the topic with the highest weights are indeed best-selling by human inspection. Of course, this would be somewhat subjective because we are the judges! Empirically, however, this is somewhat mitigated by the more refined topics selected by hyperparameter tuning that point to very specific titles that we can look up its ticket sales directly. 

### How well can the SLDA model predict box-office performance out of samples?

The bar plot below shows the percentage difference of prediction efficacy metrics (MAE, MSE, R2) in various data modes (e.g. train vs validation). The positive bars of MAE and R2 in test data both indicate that SLDA outperforms LDA out of samples, while the negative bar MSE indicates that it is underperforming. However, on the validation performance, SLDA seems to very slightly outperform LDA.

<img width="709" alt="Screen Shot 2021-04-11 at 11 55 09 PM" src="https://user-images.githubusercontent.com/9246300/114338584-5c517380-9b21-11eb-8049-d4bf4cd65e4d.png">

<img width="713" alt="Screen Shot 2021-04-12 at 12 04 26 AM" src="https://user-images.githubusercontent.com/9246300/114339212-a8e97e80-9b22-11eb-8558-2c4f2b4afb3c.png">

<img width="709" alt="Screen Shot 2021-04-12 at 12 04 46 AM" src="https://user-images.githubusercontent.com/9246300/114339236-b30b7d00-9b22-11eb-922c-f220f5bd67fb.png">




### What are the best-selling topics? Are they your favorites?
  
Best-selling movie topics found by SLDA trained with 50 topics out of 22k movie titles, with the top 10 words listed for each topic:

The best-selling topic ['benjamin', 'pitt', 'fincher', 'brad', 'freeman', 'bride', 'button', 'hudson', 'wed', 'walken'] seems to loads on Brad Pitt's classic The Curious Case of Benjamin Button (directed by David Fincher) with $335 million sales.  

The runner-up topic ['alien', 'earth', 'robot', 'space', 'sci-fi', 'shark', 'human', 'ship', 'scienc', 'planet'] is clearly referring to the Aliens series and also the Jaws series, both of which are extremely successful, with a whopping $240 million and $470 million sales in just one of their sequels, respectively.

The ranking of the movies could perhaps be switched, yet it is hard to make a judgment on that because there are many words that we haven't looked at, and the sequels sometimes could have an underperformance too like (Jaws: The Revenge with $51 million sales).


## Reference
 
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
