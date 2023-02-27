# Keywords Extraction with Semantic Search

**We perform the following steps to get the highest similar keywords to queries:**

1. Clean text
2. Extract highest frequent combination of words using n-gram
3. clean returned keywords by removing dublicates
4. Use semantic search to get the highest similar keywords for each query in queries.yaml file

---

## How to run:
  **1. install required libraries**
  
        pip install -r requiremtns.txt
        
        
  **2. Update utilities/queries.yml file with all the queries you want to look for their kewords**


  **3. run "run.py" file**

        python run.py
**the result will be saved as csv file "data/result.csv", each column is a query, and all the similar keywords are writting as a list, we can prettify that by writting each one in seperate row**


---

## Notes:

1. Senetence embeding give us in most cases beteer results, but we used here word embeding as:
  * "UBC-NLP/ARBERT" have more arabic words so it gives us better resutls.
  * We compare sentences with 1-5 words, so we don't have that long sentences that be vital to use sentence embeding with them.
  
2. We can Extract keywords using "BERTopic" instead of n-gram
