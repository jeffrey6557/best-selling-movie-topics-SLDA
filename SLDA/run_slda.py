from utils import *
from configuration import *
logger = logging.getLogger(__name__)


class tpLDAModel(object):
    def __init__(self,
                 model_type,
                 corpus,
                 param=None,
                 train_iters=1000,
                 infer_iters=100,
                 mode='train',
                 model_path=None,
                 weight_path=None,
                 **kwargs):
        self._model_type = model_type
        self._label_reader = None
        if param is None:
            param = {}
        else:
            param = param.copy()
        if model_type == 'SLDA':
            model = tp.SLDAModel
        elif model_type == 'LDA':
            param.pop('vars', None)
            model = tp.LDAModel
        else:
            raise NotImplemented('Only SLDA and LDA models are supported!')
        self._model_class = model
        self._corpus = corpus
        self._param = param
        self._model = model(corpus=corpus, **param)
        self._train_iters = train_iters
        self._infer_iters = infer_iters
        self._mode = mode
        self._model_path = model_path
        self._weight_path = weight_path
        self._kwargs = kwargs
        self._weights = None

    def train(self, mode=None):
        ''' if mode = "train" or "test", train a new model; otherwise, load a model from self._model_path'''
        if mode is None:
            mode = self._mode
        if mode in ['train', 'retrain']:
            self._train()
        else:
            self._load()
        return self

    def _train(self):
        # make LDA model and train
        self._model.train(0)
        logger.info('Num docs: {}, Vocab size: {}, Num words: {}'.format(
            len(self._model.docs), len(self._model.used_vocabs), self._model.num_words))
        logger.info('Removed top words: {}'.format(self._model.removed_top_words))

        iters_per_loop = self._train_iters
        num_loops = 1
        if self._train_iters > 100:
            iters_per_loop = 100
            num_loops = self._train_iters // iters_per_loop
        for i in range(0, num_loops, 1):
            self._model.train(iters_per_loop)
            logger.info('Iteration: {}\tLog-likelihood: {}'.format(i, self._model.ll_per_word))

        self._model.summary()

        if self._model_type == 'SLDA':
            self._weights = to_flatten(self._model.get_regression_coef())
        else:
            y = to_flatten([d.vars for d in get_labels(self._corpus)])
            # topic_dist is N x K where N is the number of documents and K is the number of topics
            topic_dist, ll = self._model.infer(self._model.docs, iter=self._infer_iters, together=False)
            self._train_set_topic_dist = np.array(topic_dist)
            has_valid_labels = ~np.isnan(y)
            X = self._train_set_topic_dist[has_valid_labels]
            self._weights = solve(X.T.dot(X), X.T.dot(y[has_valid_labels]))
        topic_ids = self._weights.argsort()[::-1]
        for k in topic_ids:
            logger.info("== Topic #{} == with weight {:.2f}".format(k, self._weights[k]))
            logger.info([w for w, _ in self._model.get_topic_words(k, top_n=5)])

        if self._model_path is not None:
            self._model.save(self._model_path)
            pd.Series(self._weights).to_csv(self._weight_path, header=False, index=False)
            logger.info('Saved trained model to {}'.format(self._model_path))
        return self

    def _load(self):
        self._model = self._model_class.load(self._model_path)
        self._weights = pd.read_csv(self._weight_path, header=None, index_col=None).values.flatten()
        logger.info('Loaded trained model from {}'.format(self._model_path))
        return self

    def predict(self, unseen_corpus=None, param=None):
        '''
        predict labels of trained or unseen documents using the trained model
        return a tuple of (array, float, list of tp.Documents) as (prediction, ll_per_word, docs)
        '''
        if 'train' in self._mode:
            docs = self._model.docs
            ll_per_word = self._model.ll_per_word
            if self._model_type == 'SLDA':
                prediction = self._model.estimate(docs)
            else:
                prediction = self._train_set_topic_dist.dot(self._weights)
        else:
            if unseen_corpus is None:
                unseen_corpus = self._corpus
            if param is None:
                param = self._param
            docs = self._model_class(corpus=unseen_corpus, **param).docs
            unseen_docs = []
            doc_lengths = []
            for d in docs:
                words = extract_words(unseen_corpus, d)
                unseen_doc = self._model.make_doc(words)
                unseen_docs.append(unseen_doc)
                doc_lengths.append(len(words))
            topic_dist, ll = self._model.infer(unseen_docs, iter=self._infer_iters, together=False)
            ll_per_word = np.mean(to_flatten(ll) / to_flatten(doc_lengths))
            logger.info('Loglikelihood per word in training: {:.2f}; in validation: {:.2f}'.format(
                self._model.ll_per_word, ll_per_word
            ))
            if self._model_type == 'SLDA':
                prediction = np.array(topic_dist).dot(self._model.get_regression_coef().T)
            else:
                prediction = np.array(topic_dist).dot(self._weights)

        return prediction, ll_per_word


def load_data(fold_idx, by, mode, valid_label_only, column_label):
    suffix = get_file_suffix(fold_idx)
    valid = ''
    if valid_label_only:
        valid = 'valid_'
    corpus_path = '{}corpus/{}/{}_{}{}_corpus{}.cleaned'.format(PROCESSED_DATA_DIR, by, mode,
                                                                valid, column_label, suffix)
    logger.info('Loaded cleaned labeled corpus from {}'.format(corpus_path))
    cleaned_corpus = tp.utils.Corpus.load(corpus_path)
    return cleaned_corpus


def run_experiments(model_type, by, mode, param, fold_idx, train_iters, infer_iters, valid_label_only, column_label,
                    overwrite=False):
    logger.info('{} {} with {} topics'.format(model_type, by, param['k']))
    # get labeled corpus
    cleaned_corpus = load_data(fold_idx, by, mode, valid_label_only, column_label)

    # train model
    suffix = get_file_suffix(**dict(fold_idx=fold_idx, k=param['k'], **base_param))
    model_path = '{}/train_model{}.bin'.format(TRAINED_MODEL_DIR, suffix)
    weight_path = '{}/train_weight{}.csv'.format(TRAINED_MODEL_DIR, suffix)
    model = tpLDAModel(model_type=model_type,
                       param=param,
                       corpus=cleaned_corpus,
                       train_iters=train_iters,
                       infer_iters=infer_iters,
                       mode=mode,
                       model_path=model_path,
                       weight_path=weight_path)
    model.train()
    # predict based on trained models
    prediction, ll_per_word = model.predict()
    # evaluate prediction accuracy of movie sales based on topic distribution per doc
    labels = [d.vars for d in get_labels(cleaned_corpus)]
    stats = compute_metrics(labels, prediction)

    # format evaluation metrics and save to csv
    save_path = '{}/eval_metrics.csv'.format(EFFICACY_DIR)
    param_to_save = param.copy()
    extra_param = dict(mode=mode, fold_idx=fold_idx, ll=ll_per_word, valid=valid_label_only)
    param_to_save.update(extra_param)
    for k, v in list(param_to_save.items()) + list(base_param.items()):
        stats[k] = v
    if os.path.exists(save_path) and not overwrite:
        existing_stats = pd.read_csv(save_path, index_col=0)
        stats = pd.concat([existing_stats, stats], ignore_index=True)
    stats.to_csv(save_path)
    return stats


if __name__ == '__main__':

    # First, we perform hyper-parameter tuning using k-fold cross-validation in these two modes:
    # 1) train: given the hyper-parameter set, train a model on the training subset from one of the k folds
    # 2) validation: apply the trained model to the validation set and evaluate the performance
    # After cross-validation, we then apply the optimal hyper-parameters in these two modes:
    # 3) retrain: retrain the model on the entire training set of the IMDb review corpus
    # 4) test: apply the model on the test set of the review corpus and evaluate performance
    # you can specify any of them in modes_to_run
    modes_to_run = ['train', 'validation', 'retrain', 'test'][:]
    # note that 'validation' presupposes 'train' has run; 'test' presupposes 'retrain'.
    # For modes 1) and 2), must specify n_splits (int) not greater than # folds you saved from process_corpus.py;
    # If you run modes 3) and 4), this will not take effect (fold is not defined).
    n_splits = 5

    # iterations to train a model or use it to infer unseen documents
    train_iters = 1000
    infer_iters = 100
    # specify the column to use as label for supervised learning:
    # available options: 'Domestic_sale', 'International_sale', and 'Worldwide_sale'
    column_label = 'Worldwide_sale'
    # document_definitions: a document is defined as a review or a collection of reviews concatenated
    # if 'review' is specified, subsequently the loss function will be weighted by number of reviews per title,
    # with the reasoning that the highly reviewed (and thus popular) title should have higher weight in sales prediction
    document_definitions = ['movie', 'review'][:1]
    # mode_types specify supervised (SLDA) or unsupervised LDA
    # unsupservised LDA will use the trained topic distribution of shape (n_train, k) to fit on the target using OLS
    model_types = ['SLDA', 'LDA']
    # valid_label_options specify whether to use documents with valid sales performance data only
    valid_label_options = [True, False][:1]

    # parameter_set is a list of dictionary that specifies the hyper-parameters
    # see the documentation on LDA and SLDA from https://bab2min.github.io/tomotopy/v0.9.0/en/
    # here, we evaluate the effect of tuning k
    parameter_set = [dict(tw=tp.TermWeight.PMI, vars='l', k=k, rm_top=3, min_cf=10, min_df=5)
                     for k in [10, 25, 50, 100, 150, 200]]

    for valid_label_only in valid_label_options:
        for document_definition in document_definitions:
            for model_type in model_types:
                base_param = dict(model_type=model_type,
                                  by=document_definition,
                                  stemmer='porter',
                                  label='Worldwide_sale',
                                  iterations=train_iters)
                for param in parameter_set:
                    for mode in modes_to_run:
                        folds = range(n_splits) if mode in ['train', 'validation'] and n_splits is not None else [None]
                        for fold_idx in folds:
                            run_experiments(model_type, document_definition, mode, param, fold_idx,
                                            train_iters, infer_iters, valid_label_only, column_label, overwrite=False)
