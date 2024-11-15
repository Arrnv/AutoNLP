import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

class EDA:
    @staticmethod
    def Word_Cloud(df, col: str):
        text_data = " ".join(df[col].astype(str).tolist())
        
        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=200
        ).generate(text_data)
        
        # Plot the word cloud
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