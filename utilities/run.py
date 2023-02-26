from utilities.keyword_extraction import *

text_processor = TextProcessor()
df = pd.read_csv("../data/Output_sorted.csv", encoding='utf-8-sig')
df = df.iloc[:100]
df = df.dropna(subset=['DownloadData'])
df['clean_text'] = df['DownloadData'].progress_apply(text_processor.process)


keywords = extract_keywords(list(df['clean_text']), text_processor.process)
print(keywords.head())