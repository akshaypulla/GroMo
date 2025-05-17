import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # Added TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation # For LDA
# from textblob import TextBlob # Already imported
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Already imported
import numpy as np # For LDA topic interpretation

# --- Configuration (same as before) ---
CSV_FILE_PATH = '/Users/akshaypulla/Desktop/GroMo/gromo_play_store_reviews_detailed.csv'
FILTER_FUTURE_DATES = True
FUTURE_DATE_THRESHOLD = pd.Timestamp.now() + pd.Timedelta(days=1)
LDA_N_TOPICS = 8 # Number of topics for LDA - EXPERIMENT WITH THIS!
LDA_N_TOP_WORDS = 10 # Number of top words to display per topic

# --- Load and Basic Preprocessing (same as before) ---
print(f"Loading data from {CSV_FILE_PATH}...")
try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {CSV_FILE_PATH}. Please check the path.")
    exit()

print("Initial DataFrame Info:")
# df.info() # Can be verbose, uncomment if needed
# print("\nFirst 5 rows:")
# print(df.head())

df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')
original_row_count = len(df)
if FILTER_FUTURE_DATES:
    future_dates_mask = df['Review_Date'] > FUTURE_DATE_THRESHOLD
    num_future_dates = future_dates_mask.sum()
    if num_future_dates > 0:
        print(f"\nWarning: Found {num_future_dates} reviews with dates beyond {FUTURE_DATE_THRESHOLD}.")
        df = df[~future_dates_mask].copy()
        print(f"Filtered out {num_future_dates} rows. New row count: {len(df)}")
df.dropna(subset=['Review_Date'], inplace=True)
print(f"Rows after handling date issues: {len(df)}")

# --- Text Preprocessing Function (same as before) ---
stop_words = set(stopwords.words('english'))
custom_stopwords = ['app', 'gromo', 'application', 'please', 'also', 'get', 'even', 'would', 'could', 'make', 'really', 'good', 'great', 'nice', 'best', 'awesome', 'worst', 'bad', 'hai', 'sir', 'mam', 'ok', 'thank'] # Added more
stop_words.update(custom_stopwords)
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# Apply preprocessing to Review_Message
if 'Review_Message' in df.columns:
    print("\nPreprocessing review messages...")
    df['Processed_Message'] = df['Review_Message'].apply(preprocess_text)
    print("Text preprocessing complete for Review_Message.")
else:
    print("Error: 'Review_Message' column not found.")

# --- Sentiment Analysis (VADER - same as before, but ensure it runs early for topic sentiment) ---
if 'Review_Message' in df.columns and 'Processed_Message' in df.columns: # Check if 'Processed_Message' exists
    print("\n--- 6. Sentiment Analysis (VADER) ---")
    analyzer = SentimentIntensityAnalyzer()
    df['VADER_Sentiment_Compound'] = df['Review_Message'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
    def categorize_sentiment(compound_score):
        if compound_score >= 0.05: return 'Positive'
        elif compound_score <= -0.05: return 'Negative'
        else: return 'Neutral'
    df['VADER_Sentiment_Label'] = df['VADER_Sentiment_Compound'].apply(categorize_sentiment)
    print("VADER sentiment analysis complete.")
else:
    print("Skipping VADER sentiment analysis as 'Review_Message' or 'Processed_Message' is not available.")


# --- 8. Topic Modeling (LDA) ---
print("\n--- 8. Topic Modeling (LDA) ---")
if 'Processed_Message' in df.columns and not df['Processed_Message'].dropna().empty:
    # Use TF-IDF for LDA
    # Consider min_df and max_df to filter out very rare or very common words if dataset is noisy
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=list(stop_words))
    try:
        tfidf = tfidf_vectorizer.fit_transform(df['Processed_Message'].dropna())
        feature_names = tfidf_vectorizer.get_feature_names_out()

        if tfidf.shape[0] > 0 and tfidf.shape[1] > 0: # Check if tfidf matrix is not empty
            print(f"Fitting LDA model with {LDA_N_TOPICS} topics...")
            lda = LatentDirichletAllocation(n_components=LDA_N_TOPICS, random_state=42, learning_method='online')
            lda.fit(tfidf)

            print(f"\nTop {LDA_N_TOP_WORDS} words per topic:")
            topic_keywords = {}
            for topic_idx, topic in enumerate(lda.components_):
                top_words_indices = topic.argsort()[:-LDA_N_TOP_WORDS - 1:-1]
                top_words = [feature_names[i] for i in top_words_indices]
                topic_keywords[f"Topic {topic_idx+1}"] = top_words
                print(f"Topic {topic_idx+1}: {', '.join(top_words)}")

            # Assign dominant topic to each review
            topic_distributions = lda.transform(tfidf)
            df.loc[df['Processed_Message'].dropna().index, 'Dominant_Topic_ID'] = topic_distributions.argmax(axis=1)
            
            # Create human-readable topic names (YOU WILL NEED TO EDIT THESE BASED ON LDA_N_TOP_WORDS OUTPUT)
            # This is a placeholder - you must interpret the topics from their keywords
            topic_id_to_name_map = {
                i: f"Topic {i+1} ({', '.join(topic_keywords.get(f'Topic {i+1}', ['N/A']))[:30]}...)"
                for i in range(LDA_N_TOPICS)
            }
            # Example (manual after inspecting topic_keywords):
            # topic_id_to_name_map = {
            #     0: "Payment & Earnings",
            #     1: "App Performance/Bugs",
            #     2: "Customer Support",
            #     3: "Product/Service Issues",
            #     4: "Ease of Use/Interface",
            #     # ... and so on for LDA_N_TOPICS
            # }
            
            df['Dominant_Topic_Name'] = df['Dominant_Topic_ID'].map(topic_id_to_name_map).fillna("Unassigned")

            print("\nDominant Topic Counts:")
            print(df['Dominant_Topic_Name'].value_counts())

            plt.figure(figsize=(10, max(6, LDA_N_TOPICS * 0.5))) # Adjust height based on number of topics
            sns.countplot(y='Dominant_Topic_Name', data=df, order=df['Dominant_Topic_Name'].value_counts().index, palette='cubehelix')
            plt.title(f'Distribution of {LDA_N_TOPICS} Dominant Topics')
            plt.xlabel('Number of Reviews')
            plt.ylabel('Dominant Topic')
            plt.tight_layout()
            plt.savefig('dominant_topic_distribution.png')
            plt.show()

        else:
            print("TF-IDF matrix is empty after vectorization. Cannot perform LDA. Check preprocessing and data.")
    except ValueError as e:
        print(f"Error during TF-IDF or LDA: {e}. This can happen if the vocabulary is empty after preprocessing.")
else:
    print("Skipping LDA Topic Modeling as 'Processed_Message' is empty or not available.")


# --- 9. Sentiment Analysis by Topic ---
print("\n--- 9. Sentiment Analysis by Topic ---")
if 'Dominant_Topic_Name' in df.columns and 'VADER_Sentiment_Compound' in df.columns:
    if df['Dominant_Topic_Name'].nunique() > 1 and df['Dominant_Topic_Name'].value_counts().max() > 1 : # Ensure there's more than one topic and some reviews per topic
        plt.figure(figsize=(12, 7))
        # Sort topics by median sentiment or by name for consistency
        sorted_topics = df.groupby('Dominant_Topic_Name')['VADER_Sentiment_Compound'].median().sort_values().index
        sns.boxplot(x='Dominant_Topic_Name', y='VADER_Sentiment_Compound', data=df, palette='Spectral', order=sorted_topics)
        plt.title('Sentiment (VADER Compound) Distribution by Dominant Topic')
        plt.xlabel('Dominant Topic')
        plt.ylabel('VADER Sentiment Compound Score (-1 to 1)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig('sentiment_by_topic.png')
        plt.show()

        # Average sentiment per topic
        avg_sentiment_by_topic = df.groupby('Dominant_Topic_Name')['VADER_Sentiment_Compound'].mean().sort_values()
        print("\nAverage VADER Sentiment Compound Score by Topic:")
        print(avg_sentiment_by_topic)
    else:
        print("Not enough distinct topics or reviews per topic to generate 'Sentiment by Topic' plot.")
else:
    print("Skipping Sentiment Analysis by Topic as 'Dominant_Topic_Name' or 'VADER_Sentiment_Compound' is not available.")


# --- 10. Refined Keyword Tracking with Context (Illustrative Example) ---
print("\n--- 10. Refined Keyword Tracking (Illustrative) ---")
if 'Review_Message' in df.columns: # Using original message for more context
    # Example: Track "payment" and see surrounding words (collocations)
    payment_reviews = df[df['Review_Message'].astype(str).str.contains("payment|payout|withdrawal", case=False, na=False)]
    
    if not payment_reviews.empty:
        print(f"Found {len(payment_reviews)} reviews mentioning 'payment', 'payout', or 'withdrawal'.")
        
        # Simple N-gram analysis around the keyword "payment" for reviews mentioning it
        # This is a basic way to get context. More advanced: dependency parsing, etc.
        payment_context_corpus = payment_reviews['Processed_Message'].dropna() # Use processed for cleaner ngrams

        if not payment_context_corpus.empty:
            try:
                # Focus on bigrams/trigrams that might include 'payment' or related terms
                # We are looking at all bigrams/trigrams in these specific reviews
                vec_payment_context = CountVectorizer(ngram_range=(2,3), stop_words=list(stop_words)).fit(payment_context_corpus)
                bag_payment_context = vec_payment_context.transform(payment_context_corpus)
                sum_payment_context = bag_payment_context.sum(axis=0)
                payment_context_freq = [(word, sum_payment_context[0, idx]) for word, idx in vec_payment_context.vocabulary_.items()]
                payment_context_freq = sorted(payment_context_freq, key = lambda x: x[1], reverse=True)
                
                print("\nTop N-grams in reviews mentioning 'payment/payout/withdrawal':")
                for ngram, freq in payment_context_freq[:15]:
                    print(f"- '{ngram}': {freq}")
            except ValueError as e:
                print(f"Could not generate n-grams for payment context: {e}")
        else:
            print("No processed text available for payment context analysis.")
    else:
        print("No reviews found mentioning 'payment', 'payout', or 'withdrawal'.")
else:
    print("Skipping Refined Keyword Tracking as 'Review_Message' is not available.")


# --- 11. Analyze Developer Reply Content (Basic) ---
print("\n--- 11. Basic Analysis of Developer Reply Content ---")
if 'Developer_Reply_Message' in df.columns:
    # Filter out rows where reply is NaN or just whitespace
    df_dev_replies = df[df['Developer_Reply_Message'].notna() & (df['Developer_Reply_Message'].str.strip() != '')].copy()

    if not df_dev_replies.empty:
        print(f"Analyzing {len(df_dev_replies)} developer replies...")
        # Preprocess developer replies
        df_dev_replies['Processed_Reply'] = df_dev_replies['Developer_Reply_Message'].apply(preprocess_text)

        # VADER Sentiment of Developer Replies
        df_dev_replies['Reply_VADER_Sentiment_Compound'] = df_dev_replies['Developer_Reply_Message'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
        df_dev_replies['Reply_VADER_Sentiment_Label'] = df_dev_replies['Reply_VADER_Sentiment_Compound'].apply(categorize_sentiment)

        print("\nDeveloper Reply Sentiment (VADER):")
        print(df_dev_replies['Reply_VADER_Sentiment_Label'].value_counts(normalize=True) * 100)

        plt.figure(figsize=(8, 5))
        sns.countplot(x='Reply_VADER_Sentiment_Label', data=df_dev_replies, palette='crest', order=['Positive', 'Neutral', 'Negative'])
        plt.title('Sentiment Distribution of Developer Replies (VADER)')
        plt.savefig('developer_reply_sentiment.png')
        plt.show()
        
        # Top N-grams in Developer Replies
        # (This shows common phrases used by developers)
        reply_corpus = df_dev_replies['Processed_Reply'].dropna()
        if not reply_corpus.empty:
            try:
                # Using the existing plot_top_ngrams function if defined earlier, or inline:
                vec_reply = CountVectorizer(ngram_range=(1,2), stop_words=list(stop_words)).fit(reply_corpus) # Uni and Bigrams
                bag_reply = vec_reply.transform(reply_corpus)
                sum_reply = bag_reply.sum(axis=0)
                reply_freq = [(word, sum_reply[0, idx]) for word, idx in vec_reply.vocabulary_.items()]
                reply_freq = sorted(reply_freq, key = lambda x: x[1], reverse=True)
                
                top_reply_ngrams_df = pd.DataFrame(reply_freq[:15], columns=['Ngram', 'Frequency'])
                
                plt.figure(figsize=(10, 7))
                sns.barplot(x='Frequency', y='Ngram', data=top_reply_ngrams_df, palette='flare')
                plt.title('Top N-grams in Developer Replies')
                plt.tight_layout()
                plt.savefig('top_ngrams_developer_replies.png')
                plt.show()
                print("\nTop N-grams in Developer Replies:")
                for ngram, freq in reply_freq[:15]:
                    print(f"- '{ngram}': {freq}")

            except ValueError as e:
                print(f"Could not generate n-grams for developer replies: {e}")
        else:
            print("No processed text available for developer reply n-gram analysis.")
    else:
        print("No developer replies found to analyze.")
else:
    print("Skipping Developer Reply Analysis as 'Developer_Reply_Message' is not available.")


# --- Final Message ---
print("\n\n--- Extended Analysis Complete ---")
print("Review the generated plots (PNG files) and console output.")
print("Remember to interpret LDA topics by examining their keywords and adjust LDA_N_TOPICS if needed.")