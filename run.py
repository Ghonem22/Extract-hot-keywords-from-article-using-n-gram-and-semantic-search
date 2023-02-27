from keyword_extraction.keyword_extraction import *
from semantic_search.semantic import *
import yaml


with open(r'utilities/queries.yml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    queries = yaml.load(file, Loader=yaml.FullLoader)


text_processor = TextProcessor()
semantic_search = SemanticSearch("UBC-NLP/ARBERT")

# read data
df = pd.read_csv("data/Output_sorted.csv", encoding='utf-8-sig')
# take sample of data for faster processing
df = df.iloc[:400]

# clean data
df = df.dropna(subset=['DownloadData'])
df['clean_text'] = df['DownloadData'].progress_apply(text_processor.process)


keywords = extract_keywords(list(df['clean_text']), text_processor.process, text_processor.arabic_stopwords)

keywords = keywords.iloc[:30]

keywords = remove_repeated_keywords(keywords)

# get word embedding of keywords
keywords['keywords_embedding'] = keywords["key_phrases"].progress_apply(lambda x: semantic_search.get_vector(x))

# get word embedding of queried
queries = pd.DataFrame(queries['keywords'], columns = ['key_phrases'])
queries['queries_embedding'] = keywords["key_phrases"].progress_apply(lambda x: semantic_search.get_vector(x))

embedings = np.array(list(keywords['keywords_embedding'])).reshape((len(keywords['keywords_embedding']), 768))

queries['keywords'] = queries["queries_embedding"].progress_apply(lambda x: semantic_search.get_most_similar_keywords(x, embedings,keywords , 10))

del queries['queries_embedding']

print(queries)