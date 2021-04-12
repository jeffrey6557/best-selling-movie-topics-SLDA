from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import KFold
from utils import *
from configuration import *
from download_sales import get_labeled_sales

logger = logging.getLogger(__name__)


def label_review_corpus(folder, subfolder, valid_label_only=True, column_label='Worldwide_sale', by_review=True):
    # file name contains review id and rating, separated by '_'
    directory = './aclImdb/{}/{}/'.format(folder, subfolder)
    logger.info('Loading raw corpus at {}, this might take several minutes ...'.format(directory))
    files_sorted = OrderedDict(sorted([(int(f.split('_')[0]), f) for f in os.listdir(directory)]))

    # get labeled sales
    labeled_sales = get_labeled_sales(folder, subfolder)
    labeled_sales[column_label] = labeled_sales[column_label].apply(np.log)

    data_feeder = []
    for idx, title in enumerate(labeled_sales['url_id'].unique()):
        is_title = labeled_sales['url_id'] == title
        label = labeled_sales.loc[is_title, column_label].unique()[0]
        review_ids = labeled_sales.loc[is_title].index.tolist()
        if np.isnan(label) and valid_label_only:
            continue
        reviews = []
        for review_id in review_ids:
            path = directory + files_sorted[review_id]
            doc = open(path, encoding='utf-8').readlines()[0]
            # treat one review as a document
            if by_review:
                data_feeder.append((doc, title, dict(y=[label])))
            else:
                reviews.append(doc)
        # concatenate all the reviews of a movie as a document
        if not by_review:
            doc = ' '.join(reviews)
            data_feeder.append((doc, title, dict(y=[label])))

        if idx % 100 == 0 and idx > 1:
            logger.info('{} titles read'.format(idx))

    return data_feeder


def get_corpus_processor(stemmer='porter', custom_stopwords=['film', 'movie']):
    stemmer = stemmer.lower()
    if stemmer == 'porter':
        stemmer = PorterStemmer()
    elif stemmer == 'lanscater':
        stemmer = LancasterStemmer()
    stops = set(stopwords.words('english')).union(custom_stopwords)
    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(stemmer=stemmer.stem),
                             stopwords=lambda x: len(x) <= 2 or x in stops)
    return corpus


def split_into_kf_cv(data_feeder, valid_label_only=True, document_definition='review', column_label='Worldwide_sale', n_splits=3, random_state=0, overwrite=True):
    '''
    take a data_feeder and split into k-folds, and save to files
    :param labeled_data: list
    :param n_splits: int
    :return:
    '''
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kf.get_n_splits(data_feeder)
    modes = ['train', 'validation']
    for fold_idx, data_indices in enumerate(kf.split(data_feeder)):
        logger.info("TRAIN: {} rows VALIDATION: {} rows".format(len(data_indices[0]), len(data_indices[1])))
        for mode, data_index in zip(modes, data_indices):
            # trick to maintain inner nested data structure while maintaining np.array fancy index slicing
            fold = list(pd.Series(data_feeder).values[data_index])
            process_labeled_data(fold, valid_label_only=valid_label_only, document_definition=document_definition, column_label=column_label, mode=mode, fold_idx=fold_idx, overwrite=overwrite)
    return


def process_labeled_data(data_feeder, valid_label_only=True, document_definition='review', column_label='Worldwide_sale', mode='test', fold_idx=None,
                         overwrite=True):
    corpus_processor = get_corpus_processor()
    corpus_processor.process(data_feeder)
    suffix = get_file_suffix(fold_idx)
    valid = ''
    if valid_label_only:
        valid = 'valid_'
    corpus_path = '{}corpus/{}/{}_{}{}_corpus{}.cleaned'.format(PROCESSED_DATA_DIR, document_definition, mode, valid, column_label, suffix)
    if overwrite or (not os.path.exists()):
        corpus_processor.save(corpus_path)
        logger.info('Saved cleaned labeled corpus to {}'.format(corpus_path))
    return


if __name__ == '__main__':

    # the IMDb datasets are organized into train and test sets, each with positive and negative reviews,
    # in addition, there is an unsupervised dataset under train/
    folders = ['train', 'test']
    subfolders = ['pos', 'neg', 'unsup']
    # n_splits controls the kinds of datasets you save:
    # if set to None, it will process the entire training corpus (for 'retrain' mode) and test set (for 'test' mode)
    # if set to k (int), it will split the training corpus into k folds for cross-validation
    n_splits = None
    # column_label specifies the column to use as target for training
    # available options are 'Domestic_sale', 'International_sale', and 'Worldwide_sale'
    column_label = 'Worldwide_sale'
    # valid_label_options: available options to use reviews with valid sales data only or not
    valid_label_options = [True, False]
    # document_definitions: a document is defined as a review or a collection of reviews concatenated
    # if 'review' is chosen, subsequently, the loss function will be weighted by number of reviews per title
    # with the reasoning that the highly reviewed (and thus popular) title should have higher weight in sales prediction
    document_definitions = ['movie', 'review']

    # Now we will process and save the corpus by
    # 1) cleaning the documents using nltk
    # 2) labeling each review/title by its sales performance
    for valid_label_only in valid_label_options:
        for document_definition in document_definitions:
            for folder in folders:
                for subfolder in subfolders:
                    # there is no a test set for unsupervised data
                    if folder == 'test' and subfolder == 'unsup':
                        continue
                    logger.info('Processing data {} {}'.format(folder, subfolder))
                    # get review corpus
                    data_feeder = label_review_corpus(folder, subfolder,
                                                      column_label=column_label, by_review=True)

                if folder == 'train' and n_splits is not None:
                    # splits into k-fold training and validation subsets
                    split_into_kf_cv(data_feeder, document_definition=document_definition,
                                     valid_label_only=valid_label_only, column_label=column_label,
                                     n_splits=n_splits, overwrite=True)
                else:
                    # save dataset for 'retrain' or 'test' mode
                    mode = folder if folder == 'test' else 'retrain'
                    process_labeled_data(data_feeder, document_definition=document_definition,
                                         valid_label_only=valid_label_only, column_label=column_label,
                                         mode=mode, overwrite=True)
    logger.info('Finish loading all data')

