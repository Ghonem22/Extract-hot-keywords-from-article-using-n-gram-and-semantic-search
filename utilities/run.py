from utilities.keyword_extraction import *

text_processor = TextProcessor()

# read data
df = pd.read_csv("../data/Output_sorted.csv", encoding='utf-8-sig')
# take sample of data for faster processing
df = df.iloc[:250]

# clean data
df = df.dropna(subset=['DownloadData'])
df['clean_text'] = df['DownloadData'].progress_apply(text_processor.process)


keywords = extract_keywords(list(df['clean_text']), text_processor.process, text_processor.arabic_stopwords)

keywords = keywords.iloc[:30]

keywords = remove_repeated_keywords(keywords)

print(keywords)