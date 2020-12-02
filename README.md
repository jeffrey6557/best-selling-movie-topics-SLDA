# best-selling-movie-topics-SLDA
Designed a method to help entertainment firms source profitable movie ideas; implemented supervised and semi-supervised LDA on IMDb movie reviews to find best-selling movie topics and predict box office performance based on a learned movie topic.

This project requires the Large Movie Review Dataset which you can download from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz. 

To run the analysis step by step:
1. Download the dataset from above and unzip the dataset at the directory '/SLDA'. 
2. run these scripts in sequence:
 - download_sales.py to scrap movie sales data from Box Office Mojo
 - preprocess_corpus.py to label and process the movie review data
 - run_experiments.py to run experiments on various settings
 - analysis.py to do exploratory analysis and visualize the prediction results
 
Best-selling movie topics found by SLDA trained with 50 topics out of 22k movie titles, with the top 10 words listed for each topic:

The best-selling topic ['benjamin', 'pitt', 'fincher', 'brad', 'freeman', 'bride', 'button', 'hudson', 'wed', 'walken'] seems to loads on Brad Pitt's classic The Curious Case of Benjamin Button (directed by David Fincher) with $335 million sales.  

The runner-up topic ['alien', 'earth', 'robot', 'space', 'sci-fi', 'shark', 'human', 'ship', 'scienc', 'planet'] is clearly referring to the Aliens series and also the Jaws series, both of which are extremely successful, with a whopping $240 million and $470 million sales in just one of their sequels, respectively.

The ranking of the movies could perhaps be switched, yet it is hard to make a judgment on that because there are many words that we haven't looked at, and the sequels sometimes could have an underperformance too like (Jaws: The Revenge with $51 million sales).


## Reference
 
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
