# best-selling-movie-topics-SLDA
Designed a method to help entertainment firms source profitable movie ideas; implemented supervised and semi-supervised LDA on IMDb movie reviews to find best-selling movie topics and predict box office performance based on a learned movie topic.

This project requires the Large Movie Review Dataset which you can download from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz. 

To run the analysis step by step:
1. Download the dataset from above and unzip the dataset at the directory /SLDA. 
2. run these scripts in sequence:
 - download_sales.py to scrap movie sales data from Box Office Mojo
 - preprocess_corpus.py to label and process the movie review data
 - run_experiments.py to run experiments on various settings
 - analysis.py to do exploratory analysis and visualize the prediction results
 
Best-selling movie topics found by SLDA:

The best-selling topic ['benjamin', 'pitt', 'fincher', 'brad', 'freeman'] with 335 million dollar sales.  

The runner-up topic ['alien', 'earth', 'robot', 'space', 'sci-fi'] with 240 million sales in one of its best sequels.


## Reference
 
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
