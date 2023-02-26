from sklearn.feature_extraction.text import TfidfVectorizer
import arabicstopwords.arabicstopwords as stp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import string
import re
from nltk.corpus import stopwords
import sklearn
import time
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score, f1_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from tqdm import tqdm
tqdm.pandas()
import yaml
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
import arabicstopwords.arabicstopwords as stp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import string
import re
from nltk.corpus import stopwords
import sklearn
import time
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score, f1_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from tqdm import tqdm
tqdm.pandas()
import yaml
import warnings
from urllib.parse import urlparse



warnings.filterwarnings("ignore")

nltk.download('stopwords')
ar_stopwords = set(stopwords.words('arabic'))


def remove_special(text):
    for letter in '#.][!XR':
        text = text.replace(letter, '')
    return text


def remove_punctuations(text):
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("گ", "ك", text)
    return text


def clean_str(text):
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, "", text)
    # #remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)

    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    # trim
    text = text.strip()

    return text


def keep_only_arabic(text):
    return re.sub(r'[a-zA-Z?]', '', text).strip()


def convert_words(text, converter):
    for w in converter:
        text = text.replace(w, converter[w])
    return text


def remove_emojis(data):
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)


def preprocess_text(text):
    # Replace @username with empty string
    text = re.sub('@[^\s]+', ' ', text)
    text = remove_emojis(text)
    # Convert www.* or https?://* to " "
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)

    # Replace #word with word
    text = re.sub(r'#([^\s]+)', r'\1', text)
    # remove punctuations
    text = remove_punctuations(text)

    # normalize the text
    text = normalize_arabic(text)

    # remove special letters
    text = remove_special(text)

    # Clean/Normalize Arabic Text
    text = clean_str(text)

    # remove english words
    text = keep_only_arabic(text)
    # stemming
    # text= stemmer.stem(text)
    if not text:
        text = ' '
    return text.strip()


df = pd.read_csv("data/Output_sorted.csv", encoding='utf-8-sig')
df = df.iloc[:100]
df = df.dropna(subset=['DownloadData'])
df['clean_text']=df['DownloadData'].progress_apply(preprocess_text)


normalize_corpus = np.vectorize(preprocess_text)
norm_corpus = normalize_corpus(list(df['clean_text']))

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(2, 5), min_df=0.0005, max_df=0.9995, stop_words=ar_stopwords)
cv_matrix = cv.fit_transform(norm_corpus)
df_bow_sklearn = pd.DataFrame(cv_matrix.toarray(),columns=cv.get_feature_names())
df = df_bow_sklearn.sum(axis = 0)
df =df.to_frame()
df.rename(columns = {0:'count'}, inplace = True)
df.sort_values(by=['count'], ascending=False, inplace = True)

