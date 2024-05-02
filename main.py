import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download('stopwords')

text = "This is an example sentence with some stopwords that we want to remove."
stop_words = set(stopwords.words('english'))

tokens = word_tokenize(text)
filtered_text = [word for word in tokens if word.lower() not in stop_words]

print(filtered_text)
