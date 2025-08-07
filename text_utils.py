from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer as pst
nltk.download('stopwords')
nltk.download('punkt_tab')
def optimized_text_processing(text):
  stopword=stopwords.words('english')
  ps=pst()
  text=text.lower()
  temp=[]
  for i in nltk.word_tokenize(text):
    if not i.isalnum():
      continue
    if i in stopword:
      continue
    temp.append(ps.stem(i))
  return " ".join(temp)