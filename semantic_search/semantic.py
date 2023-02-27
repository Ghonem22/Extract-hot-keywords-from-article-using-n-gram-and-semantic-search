from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearch:
    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_vector(self, txt):
        txt = " ".join(txt.split()[:511])
        input_ids = self.tokenizer.encode(txt, return_tensors='pt')
        output = self.model(input_ids)[0]
        return output.mean(axis=1)[0].detach().numpy().reshape(1, -1)

    def get_similarity(self, text1, text2):
        doc1 = self.get_vector(text1)
        doc2 = self.get_vector(text2)
        return round(float(cosine_similarity(doc1, doc2)), 3)

    def get_min_indices(self, lst, n):
        sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i])[:n]
        return [i for i in range(len(lst)) if i in sorted_indices]

    def get_most_similar_keywords(self, vector, vectors, topics, n):
        distances = distance.cdist(vector, vectors, "cosine")[0]
        indices = self.get_min_indices(distances, n)
        keywords = {}
        for index in indices:
            # max_similarity = 1 - distance
            topic = topics.iloc[index]['key_phrases']
            keywords[topic] = distances[index]

        keywords = sorted(keywords.items(), key=lambda x: x[1])

        return  [k[0] for k in keywords]
