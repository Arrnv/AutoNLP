import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class Preprocessor:
    def __init__(self, remove_stopwords=True, lowercase=True, keep_emojis=False):
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.keep_emojis = keep_emojis
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        if not self.keep_emojis:
            text = re.sub(r'[^\w\s]', '', text)
        if self.lowercase:
            text = text.lower()
        return text

    def tokenize_and_lemmatize(self, text):
        words = word_tokenize(text)
        if self.remove_stopwords:
            words = [word for word in words if word not in self.stop_words]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return words
