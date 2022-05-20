import os

root = os.path.dirname(__file__).replace('\\', '/')
if root[-1] != '/':
    root = root + '/'

# directories
src = root + 'src/'
models = root + 'models/'
reuters_original_dataset = '/home/antonio/nltk_data/corpora/reuters'
the20news_original_dataset = '/home/antonio/corpora/20news-18828/'

# files
reuters_dataset = '/mnt/d/Datasets/reuters-21578.csv'
the20news_dataset = '/mnt/d/Datasets/the20news.csv'
