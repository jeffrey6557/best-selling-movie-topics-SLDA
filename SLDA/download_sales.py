from bs4 import BeautifulSoup
import requests
from utils import *
from configuration import *


def clean_sales_field(df):
    sales = df.columns[df.columns.str.contains('_sale')].tolist()
    # there is no Worldwide_percent column
    percents = df.columns[df.columns.str.contains('_percent')].tolist() + [None]
    for sale, percent in zip(sales, percents):
        df[sale] = pd.to_numeric(df[sale].str.strip('$').str.replace(',', ''), errors='coerce', downcast='float')
        if percent is not None:
            df[percent] = pd.to_numeric(df[percent].str.strip('()%'), errors='coerce', downcast='float')
    return df


def load_url(url, num_retries=5):
    req = requests.get(url)
    soup = BeautifulSoup(req.content, "html.parser")

    # if we dont have legit content, request again
    attempts = 0
    while req.status_code == "404" and attempts < num_retries:
        print('An error occurs; retry for the {}th time...'.format(attempts))
        time.sleep(0.5)
        req = requests.get(url)
        soup = BeautifulSoup(req.text, "html.parser")
        attempts += 1
    return soup


def parse_sales(sales_url, suffixes, row=None):
    if row is None:
        row = {}
    soup = load_url(sales_url)
    # get gross sales
    perf = soup.find("div", class_='a-section a-spacing-none mojo-performance-summary-table')
    perf = perf.findAll('div', class_='a-section a-spacing-none')
    for tag in perf:
        category_info = tag.find('span', class_='a-size-small').text.strip().split(' ')
        category = category_info[0]
        percent = None
        if len(category_info) == 2:
            percent = category_info[1]
        sales = tag.find('span', class_='money')
        if sales is not None:
            sales = sales.text.strip()
        row[category + suffixes[0]] = sales
        row[category + suffixes[1]] = percent

    return row


def scrap_movie_sales(folder, subfolder, overwrite=False):
    # assume the IMDb dataset lives in the current work directory
    save_path = '{}{}_{}_raw_sales.csv'.format(SALES_DATA_DIR, folder, subfolder)
    if overwrite or (not os.path.exists(save_path)):
        url_path = "./aclImdb/%s/urls_%s.txt" % (folder, subfolder)
        suffixes = '_sale,_percent'.split(',')
        categories = 'Domestic,International'.split(',')
        columns = [c+s for s in suffixes for c in categories] + ['Worldwide']
        urls = pd.read_csv(url_path, names=['url'])
        unique_urls = urls['url'].unique()
        print('Start fetching sales_data from {} urls'.format(len(unique_urls)))
        rows = []
        for i, url in enumerate(unique_urls):
            url_id = re.findall('tt[0-9]+', url)[0]
            # test case: title_id = 'tt0101640', 'tt0499549'
            sales_url = 'https://www.boxofficemojo.com/title/{}/?ref_=bo_tt_ti'.format(url_id)

            try:
                row = parse_sales(sales_url, suffixes, row=None)
            except Exception as e:
                print(e)
                print('ITER {}: url id {} failed!'.format(i, url_id))
                row = dict(zip(columns, [None]*len(columns)))
            row['url_id'] = url_id
            row = pd.DataFrame(row, index=[0])
            if i % 1000 == 0 or i == 10:
                print('Finished getting {} reviews'.format(i))

            rows += [row]
        df = pd.concat(rows, ignore_index=True, sort=False)
        try:
            df.to_csv(save_path)
        except:
            print('failed to save.')
    else:
        df = read_raw_sales(folder, subfolder)
    return df


def get_labeled_sales(folder, subfolder):
    # assume the IMDb dataset lives in the current work directory
    sales = read_cleaned_sales(folder, subfolder)
    url_path = './aclImdb/{}/urls_{}.txt'.format(folder, subfolder)
    urls = pd.read_csv(url_path, names=['url'])
    urls['url_id'] = urls['url'].str.extract('(tt[0-9]+)')
    urls = urls.rename_axis('review_id').reset_index()
    labeled_sales = urls.merge(sales, on='url_id', how='left').set_index('review_id').sort_index()
    return labeled_sales


def read_raw_sales(folder, subfolder):
    read_path = '{}{}_{}_raw_sales.csv'.format(SALES_DATA_DIR, folder, subfolder)
    df = pd.read_csv(read_path, index_col=0)
    return df


def read_cleaned_sales(folder, subfolder):
    read_path = '{}{}_{}_cleaned_sales.csv'.format(SALES_DATA_DIR, folder, subfolder)
    df = pd.read_csv(read_path, index_col=0)
    return df


def save_cleaned_sales(folder, subfolder, df):
    save_path = '{}{}_{}_cleaned_sales.csv'.format(SALES_DATA_DIR, folder, subfolder)
    df.to_csv(save_path)
    return df


if __name__ == '__main__':
    # scrap movie mojo webpage for sales data for movies in the imdb dataset
    folders = ['train', 'test']
    subfolders = ['pos', 'neg']
    settings = product(folders, subfolders)
    for folder, subfolder in settings:
        overwrite = True
        if folder == 'train' and subfolder == 'unsup':
            overwrite = False
        raw_sales = scrap_movie_sales(folder, subfolder, overwrite=overwrite)
        cleaned_sales = clean_sales_field(raw_sales)
        save_cleaned_sales(folder, subfolder, cleaned_sales)