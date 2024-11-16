import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

stop_words = set(stopwords.words('english'))
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('brown')


class EDA:
    @staticmethod
    def Word_Cloud(df, col: str):
        text_data = " ".join(df[col].astype(str).tolist())
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=200
        ).generate(text_data)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for Column: {col}", fontsize=16)
        plt.show()
    
    @staticmethod
    def top_10_Tokens(df, token, top_n=10):
        all_token = [token for token_list in df[token] for token in token_list]
        token_counts = Counter(all_token)
        
        top_token = token_counts.most_common(top_n)
        tokens, frequency = zip(*top_token)
        
        plt.figure(figsize=(10,6))
        plt.bar(tokens, frequency, color='skyblue')
        
        
        plt.xlabel('Tokens', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Top {top_n} Tokens by Frequency', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        plt.show()
        
    
    @staticmethod
    def Pos_wordcloud(df, col: str):
        all_tokens = [token for token_list in df[col] for token in token_list]
        with open('en-positive-words.txt', 'r') as pos:
            poswords = pos.read().split('\n')
            
        pos_tokens = " ".join([w for w in all_tokens if w in poswords])
        plt.figure(figsize=(10,6))
        WordCloudpos = WordCloud(
            background_color='white',
            width=800,
            height=400,
            colormap='viridis',
            max_words=200
        ).generate(pos_tokens)
        plt.title("Possitive word cloud")
        plt.imshow(WordCloudpos,interpolation='bilinear')
        plt.axis("off")
        plt.grid(False)
        plt.show()
        
    @staticmethod
    def neg_wordcloud(df, col: str):
        all_tokens = [token for token_list in df[col] for token in token_list]
        with open('en-negative-words.txt', 'r') as pos:
            poswords = pos.read().split('\n')
            
        neg_tokens = " ".join([w for w in all_tokens if w in poswords])
        plt.figure(figsize=(10,6))
        WordCloudpos = WordCloud(
            background_color='white',
            width=800,
            height=400,
            colormap='viridis',
            max_words=200
        ).generate(neg_tokens)
        plt.title("negative word cloud")
        plt.imshow(WordCloudpos,interpolation='bilinear')
        plt.axis("off")
        plt.grid(False)
        plt.show()
        

    @staticmethod
    def polarity(df, col):
        df[col] = df[col].fillna("").astype(str)
        
        pattern = "[^A-Za-z.]+" 
        
        df["cleaned_text"] = df[col].apply(lambda x: re.sub(pattern, " ", x).lower())
        sentences = df["cleaned_text"].str.split(".").explode().reset_index(drop=True)
        data = pd.DataFrame({"english_text": sentences})
        print(data)
        data = data[data["english_text"].str.strip() != ""]
        
        data["number_of_words"] = data["english_text"].apply(lambda x: len(TextBlob(x).words))
        
        wh_words = {"why", "who", "which", "what", "where", "when", "how"}
        data["are_wh_words_present"] = data["english_text"].apply(
            lambda x: True if len(set(TextBlob(x).words).intersection(wh_words)) > 0 else False
        )
        
        data["polarity"] = data["english_text"].apply(lambda x: TextBlob(x).sentiment.polarity)
        data["subjectivity"] = data["english_text"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        
        average_wh_presence = data["are_wh_words_present"].mean() * 100
        average_polarity = data["polarity"].mean()
        average_subjectivity = data["subjectivity"].mean()
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(data["polarity"], bins=20, color='lightgreen', alpha=0.7)
        plt.title("Polarity Distribution")
        plt.xlabel("Polarity Score")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        plt.hist(data["subjectivity"], bins=20, color='lightcoral', alpha=0.7)
        plt.title("Subjectivity Distribution")
        plt.xlabel("Subjectivity Score")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()
        return pd.DataFrame({
            "Metric": ["Average Percentage of Sentences with WH Words", "Average Polarity", "Average Subjectivity"],
            "Value": [average_wh_presence, average_polarity, average_subjectivity]
        })
        
    @staticmethod  
    def POS_tagging_analysis(df):
        def pos_analysis(text):
            blob = TextBlob(text)
            pos_counts = Counter(tag for _, tag in blob.tags)
            return dict(pos_counts)

        df["pos_counts"] = df["processed_text"].apply(pos_analysis)
        pos_df = pd.DataFrame(df["pos_counts"].tolist()).fillna(0)
        pos_df.sum().plot(kind="bar", figsize=(10, 6), color='teal')
        plt.title("POS Tag Distribution")
        plt.xlabel("POS Tags")
        plt.ylabel("Frequency")
        plt.show()
    

    @staticmethod
    def Keyword_Extraction(df):
        df["keywords"] = df["processed_text"].apply(lambda x: TextBlob(x).noun_phrases)
        all_keywords = [kw for kws in df["keywords"] for kw in kws]
        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(20)
        plt.figure(figsize=(10,6))
        plt.barh(*zip(*reversed(top_keywords)), color="purple")
        plt.title("Top 20 Keywords")
        plt.xlabel("Frequency")
        plt.show()

    @staticmethod
    def Stopword_Analysis(df):
        df["stopword_count"] = df["processed_text"].apply(lambda x: len([ word for word in x.split() if word in stop_words]))
        plt.hist(df["stopword_count"], bins=20, color='orange', alpha=0.7)
        plt.title("Stopword Frequency Distribution")
        plt.xlabel("stopword count")
        plt.ylabel("frequency")
        plt.show()
    
    @staticmethod
    def Bigram_and_Trigram_Analysis(df):
        def plot_ngrams(corpus, ngram_range, top_n=20):
            vectorizer = CountVectorizer(ngram_range=ngram_range).fit(corpus)
            ngrams = vectorizer.transform(corpus)
            ngram_counts = ngrams.sum(axis=0)
            ngram_freq = [(ngram, ngram_counts[0, idx]) for ngram, idx in vectorizer.vocabulary_.items()]
            

            sorted_ngrams = sorted(ngram_freq, key=lambda x: x[1], reverse=True)[:top_n]
            
            ngram_labels, counts = zip(*sorted_ngrams)
            
            plt.figure(figsize=(10, 6))
            plt.barh(ngram_labels, counts, color="cyan")
            plt.title(f"Top {top_n} {' '.join(map(str, ngram_range))}-grams")
            plt.xlabel("Frequency")
            plt.ylabel("N-grams")
            plt.gca().invert_yaxis()  
            plt.show()


        plot_ngrams(df["processed_text"], (2, 2))  
        plot_ngrams(df["processed_text"], (3, 3))  
