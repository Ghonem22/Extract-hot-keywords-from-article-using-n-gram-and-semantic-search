import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from keyword_extraction.keyword_extraction import TextProcessor, extract_keywords, remove_repeated_keywords
from semantic_search.semantic import SemanticSearch


class KeywordSimilarity:
    def __init__(self, data_path, queries_path):
        self.text_processor = TextProcessor()
        self.semantic_search = SemanticSearch("UBC-NLP/ARBERT")
        self.data = pd.read_csv(data_path, encoding='utf-8-sig').iloc[:400]
        self.queries = self.load_queries(queries_path)

    def load_queries(self, queries_path):
        with open(queries_path, encoding='utf-8-sig') as file:
            queries = yaml.load(file, Loader=yaml.FullLoader)
        return pd.DataFrame(queries['queries'], columns=['key_phrases'])

    def preprocess_data(self):
        self.data = self.data.dropna(subset=['DownloadData'])
        self.data['clean_text'] = self.data['DownloadData'].progress_apply(self.text_processor.process)

    def extract_keywords(self):
        keywords = extract_keywords(list(self.data['clean_text']), self.text_processor.process,
                                    self.text_processor.arabic_stopwords)

        # here we used the highest 100 keywords for faster runneing
        keywords = keywords.iloc[:100]
        keywords = remove_repeated_keywords(keywords)
        self.keywords = keywords

    def get_embeddings(self):
        self.keywords['keywords_embedding'] = self.keywords["key_phrases"].progress_apply(
            lambda x: self.semantic_search.get_vector(x))
        self.queries['queries_embedding'] = self.queries["key_phrases"].progress_apply(
            lambda x: self.semantic_search.get_vector(x))
        self.embeddings = np.array(list(self.keywords['keywords_embedding'])).reshape(
            (len(self.keywords['keywords_embedding']), 768))

    def get_similarity(self, num_similar=10):
        self.queries['keywords'] = self.queries["queries_embedding"].progress_apply(
            lambda x: self.semantic_search.get_most_similar_keywords(x, self.embeddings, self.keywords, num_similar))
        del self.queries['queries_embedding']
        return self.queries

    def run(self, num_similar):
        self.preprocess_data()
        self.extract_keywords()
        self.get_embeddings()
        return self.get_similarity(num_similar)


if __name__ == "__main__":
    ks = KeywordSimilarity(data_path="data/Output_sorted.csv", queries_path="utilities/queries.yml")
    similarity = ks.run(num_similar = 2)
    print(similarity)


