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

def time_convert(desc, sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Time Lapsed {0}= {1}:{2}:{3}".format(desc, int(hours),int(mins),sec))