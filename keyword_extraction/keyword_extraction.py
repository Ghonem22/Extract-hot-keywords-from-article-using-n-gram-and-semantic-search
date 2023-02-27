import pandas as pd
import numpy as np
import re
import string
import arabicstopwords.arabicstopwords as stp
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
import nltk
from tqdm import tqdm
import warnings
from urllib.parse import urlparse
from nltk.stem.isri import ISRIStemmer


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
tqdm.pandas()
warnings.filterwarnings("ignore")
stemmer = ISRIStemmer()




class TextProcessor:

    def __init__(self, arabic_stopwords=stopwords.words('arabic')):
        self.arabic_stopwords = self.get_stop_words()
        self.arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''


    def get_stop_words(self):
        ar_stopwords = set(stopwords.words('arabic'))
        ar_stopwords.update([
            "مع", "من", "إلى", "في", "فى", "كان", "على", "علي", "يا", "اذا", "مش", "انه", "انا", "شي", "انت", "ان",
            "اله", "الي", "اي", "مو"
            , "بسب", "ال", "ده", "شيء", "انهم", "زي", "او", "واله", "يعني", "عشان"])

        ar_stopwords = list(ar_stopwords)

        return ar_stopwords

    def remove_special(self, text):
        for letter in '#.][!XR':
            text = text.replace(letter, '')
        return text

    def remove_punctuations(self, text):
        punctuations_list = self.arabic_punctuations + string.punctuation
        translator = str.maketrans('', '', punctuations_list)
        return text.translate(translator)

    def normalize_arabic(self, text):
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("گ", "ك", text)
        return text

    def clean_str(self, text):
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

    def keep_only_arabic(self, text):
        return re.sub(r'[a-zA-Z?]', '', text).strip()

    def convert_words(self, text, converter):
        for w in converter:
            text = text.replace(w, converter[w])
        return text

    def remove_emojis(self, text):
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
        return re.sub(emoj, '', text)


    def process(self, text):
        # Replace @username with empty string
        text = re.sub('@[^\s]+', ' ', text)
        text = self.remove_emojis(text)
        # Convert www.* or https?://* to " "
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)

        # Replace #word with word
        text = re.sub(r'#([^\s]+)', r'\1', text)
        # remove punctuations
        text = self.remove_punctuations(text)

        # normalize the text
        text = self.normalize_arabic(text)

        # remove special letters
        text = self.remove_special(text)

        # Clean/Normalize Arabic Text
        text = self.clean_str(text)

        # remove english words
        text = self.keep_only_arabic(text)
        # stemming
        # text= stemmer.stem(text)
        if not text:
            text = ' '
        return text.strip()

def extract_keywords(data, processor, ar_stopwords):


    # normalize corpus
    normalize_corpus = np.vectorize(processor)
    norm_corpus = normalize_corpus(data)

    # vectorize with CountVectorizer
    cv = CountVectorizer(ngram_range=(2, 5), min_df=0.0005, max_df=0.9995, stop_words=ar_stopwords)
    cv_matrix = cv.fit_transform(norm_corpus)
    df_bow_sklearn = pd.DataFrame(cv_matrix.toarray(), columns=cv.get_feature_names_out())

    # count word occurrences and sort
    df = df_bow_sklearn.sum(axis=0)
    df = df.to_frame()
    df.rename(columns={0: 'count'}, inplace=True)
    df.sort_values(by=['count'], ascending=False, inplace=True)
    df['key_phrases'] = df.index
    df = df.reset_index(level=0)
    del df['index']

    return df

# TODO: Optimize this function to become faster
def remove_repeated_keywords(df):
    # Create an empty dictionary to store the unique key phrases
    result = {}

    # Iterate through each row in the dataframe
    for i, row in df.iterrows():
        # Create a boolean mask to filter out the current row
        mask = df.index != i

        # Filter the dataframe to only include rows that are not the current row
        search_df = df.loc[mask]

        # Check if the current key phrase exists in any of the other key phrases
        add = True
        l = len(row['key_phrases'])
        for phrase in set(search_df['key_phrases']):
            if row['key_phrases'] in phrase and len(phrase) > l:
                add = False
                break

        # If the current key phrase is unique, add it to the result dictionary
        if add:
            result[row['key_phrases']] = row['count']

    # Convert the result dictionary to a dataframe and return it
    return pd.DataFrame(list(result.items()), columns=['key_phrases', 'count'])


