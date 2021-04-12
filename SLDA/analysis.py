import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils import *
from configuration import *
from download_sales import get_labeled_sales
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 5000)

if __name__ == '__main__':
    # exploratory analysis on the IMDb movie review datasets and sales, and save to PDF
    # first, look at the distribution of reviews per title
    pdf_page = PdfPages(PLOTS_DIR + '/movie_datasets_training_sample_distribution.pdf')
    # pick training set to visualize
    folder = 'train'
    subfolder = 'unsup'
    sales = get_labeled_sales(folder, subfolder)
    sales_columns = sales.columns[sales.columns.str.contains('sale')]
    # plot the coverage of dollar sales data by review and by title
    reviews_per_title = sales.groupby('url_id').size()
    plt.figure(figsize=(8, 5))
    reviews_per_title.hist(bins=100, density=True)
    plt.title('Distribution of reviews per title')
    # plt.legend()
    pdf_page.savefig()
    plt.close()

    # plot the coverage of dollar sales data by review and by title
    valid_sales_by_review = sales[sales_columns].notnull().astype(float)
    for c in sales_columns:
        plt.figure(figsize=(8, 5))
        valid_sales_by_review[c].hist(bins=100, density=True)
        plt.title('Valid movie {} by title'.format(c))
        pdf_page.savefig()
        plt.close()

    valid_sales_by_title = sales.groupby('url_id')[sales_columns].apply(lambda x: x.notnull().any()).astype(float)
    for c in sales_columns:
        plt.figure(figsize=(8, 5))
        valid_sales_by_title[c].hist(bins=100, density=True)
        plt.title('Valid movie {} by review'.format(c))
        pdf_page.savefig()
        plt.close()

    # plot the distribution of dollar sales by review and by title
    # use log scale as the dollar distribution is very skewed with a fat right tail:
    # some blockbusters hit hundred millions in dollar sales!
    sales_by_review = sales[sales_columns].apply(np.log)
    for c in sales_columns:
        plt.figure(figsize=(8, 5))
        sales_by_review[c].hist(bins=100)
        plt.axvline(sales_by_review[c].median(), label='median', color='red', alpha=.5)
        plt.axvline(sales_by_review[c].mean(), label='mean', color='black', alpha=0.2)
        plt.title('Movie {} by review in log US dollars'.format(c))
        plt.legend()
        pdf_page.savefig()
        plt.close()

    sales_by_title = sales.groupby('url_id')[sales_columns].first().apply(np.log)
    for c in sales_columns:
        plt.figure(figsize=(8, 5))
        sales_by_title[c].hist(bins=100)
        plt.axvline(sales_by_title[c].median(), label='median', color='red', alpha=.5)
        plt.axvline(sales_by_title[c].mean(), label='mean', color='black', alpha=0.2)
        plt.title('Movie {} by title in log US dollars'.format(c))
        plt.legend()
        pdf_page.savefig()
        plt.close()

    pdf_page.close()


    # define various settings to compare efficacy
    metrics = ['mae', 'mse', 'r2']
    colors = ['blue', 'red', 'yellow', 'black']
    model_types = ['SLDA', 'LDA']
    save_path = '{}eval_metrics.csv'.format(EFFICACY_DIR)
    # specify data modes to plot from ['train', 'validation', 'retrain', 'test']
    # define training modes to pick optimal hyperparameter (e.g. k, the number of topics)
    training_data_modes = ['train', 'validation']
    # define testing modes to evaluate out-of-sample
    testing_data_modes = ['retrain', 'test']
    # whether the experiment uses valid labels only; default is True
    valid_labels_only = True
    # we evaluate the data both by concatenating reviews of a title or by review
    document_definitions = ['movie', 'review'][:1]

    # read in and process efficacy stats
    keys = ['k',  'model_type', 'by', 'mode']
    stats = pd.read_csv(save_path, index_col=0)
    valid_label_filter = stats['valid'].astype(bool).fillna(False)
    if not valid_labels_only:
        valid_label_filter = ~valid_label_filter
    stats['k'] = stats['k'].astype(int)
    eval_stats = stats[valid_label_filter].groupby(keys)[metrics].mean()

    # evaluate the effect of k (number of topics) to pick the optimal hyperparameter
    # the criteria are the following (in order of preference to control overfitting):
    # 1) the generalization gap between training and validation to be small; and
    # 2) validation error to be small
    # if they are conflicting, we should pick the one with the best bias-variance tradeoff
    # plot evaluation stats and save to PDF
    pdf_page = PdfPages(PLOTS_DIR + './movie_sales_training_performance.pdf')
    for document_definition in document_definitions:
        document_label = 'title' if document_definition == 'movie' else document_definition
        for model_type in model_types:
            for metric in metrics:
                index = pd.IndexSlice[:, model_type, document_definition, training_data_modes]
                df_plot = eval_stats.loc[index, metric]\
                    .droplevel([1, 2])\
                    .unstack()
                title = '{} of {} movie sales prediction by {} {}s'.format(
                    metric, model_type, ' vs '.join(training_data_modes), document_label)
                df_plot.plot.bar(title=title, color=colors)
                plt.xticks(rotation=0)
                plt.xlabel('number of topics')
                plt.legend(bbox_to_anchor=(.5, -0.15), loc='upper center', ncol=4)
                plt.tight_layout()
                pdf_page.savefig()
                plt.close()
    pdf_page.close()

    # for LDA, there is a clear sweet spot at k = 50 that achieves the smallest generalization gap
    # however, for SLDA, performance is completely monotonic in both training and validation set
    # for fair comparison given the similar methodology, we pick k = 50 to compare out-of-sample performance
    best_k = 50
    # SLDA should come first for consistency
    ordered_models = ['SLDA', 'LDA']
    ordered_mode_groups = ['train data mode', 'test data mode']
    mode_tuples = zip(ordered_mode_groups, [training_data_modes, testing_data_modes])
    mode_tuples = [(g, m) for g, modes in mode_tuples for m in modes]
    # combine training and testing data modes to compare generalization gap
    data_modes = training_data_modes + testing_data_modes
    pdf_page = PdfPages(PLOTS_DIR + './movie_sales_test_performance.pdf')
    for document_definition in document_definitions:
        document_label = 'title' if document_definition == 'movie' else document_definition
        for metric in metrics:
            # compare percentage difference because magnitude of change is tiny
            # the smaller the better
            diff = []
            for model_type in ordered_models:
                index = pd.IndexSlice[best_k, model_type, document_definition, data_modes]
                model_perf = eval_stats.loc[index, metric].droplevel([0, 1, 2])
                model_perf.index = pd.MultiIndex.from_tuples(mode_tuples)
                diff += [model_perf]
            slda, lda = diff
            df_plot = (lda - slda) / slda
            df_plot = df_plot.unstack().loc[ordered_mode_groups]
            title = 'Percentage difference of {} (higher means SLDA is better)\nin movie sales prediction by {}s'.format(
                metric, document_label)
            df_plot.plot.bar(title=title, color=colors)
            plt.axhline(0)
            plt.xticks(plt.xticks()[0], ordered_mode_groups, rotation=0)
            plt.tight_layout()
            pdf_page.savefig()
            plt.close()
    pdf_page.close()

    # conclusion: we saw a very slight advantage in SLDA since adding more information helps.

