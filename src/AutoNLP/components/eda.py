import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
from textblob import TextBlob

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
        plt.subplot(1, 2, 1)
        plt.hist(data["polarity"], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title("Polarity Distribution", fontsize=14)
        plt.xlabel("Polarity", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        
        plt.subplot(1, 2, 2)
        plt.hist(data["subjectivity"], bins=20, color='salmon', edgecolor='black', alpha=0.7)
        plt.title("Subjectivity Distribution", fontsize=14)
        plt.xlabel("Subjectivity", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        
        plt.tight_layout()
        plt.show()
        average_wh_presence = data["are_wh_words_present"].mean() * 100
        average_polarity = data["polarity"].mean()
        average_subjectivity = data["subjectivity"].mean()
        
        return pd.DataFrame({
            "Metric": ["Average Percentage of Sentences with WH Words", "Average Polarity", "Average Subjectivity"],
            "Value": [average_wh_presence, average_polarity, average_subjectivity]
        })
