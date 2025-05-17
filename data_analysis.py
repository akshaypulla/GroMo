import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger') # For TextBlob POS tagging
nltk.download('vader_lexicon') # For VADER sentiment


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob # For simpler sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer # For VADER

# --- Configuration ---
CSV_FILE_PATH = '/Users/akshaypulla/Desktop/GroMo/gromo_play_store_reviews_detailed.csv' # Or your actual file name
# Handle the 'future dates' issue for "A Google user" if significant
# For now, we'll convert to datetime, coercing errors, then decide how to filter
FILTER_FUTURE_DATES = True
FUTURE_DATE_THRESHOLD = pd.Timestamp.now() + pd.Timedelta(days=1) # Dates beyond tomorrow

# --- Load and Basic Preprocessing ---
print(f"Loading data from {CSV_FILE_PATH}...")
try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {CSV_FILE_PATH}. Please check the path.")
    exit()

print("Initial DataFrame Info:")
df.info()
print("\nFirst 5 rows:")
print(df.head())

# Convert 'Review_Date' to datetime objects
# Coerce errors will turn unparseable dates into NaT (Not a Time)
df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')

# Handle potential 'future dates' from "A Google user"
original_row_count = len(df)
if FILTER_FUTURE_DATES:
    # Identify and optionally filter out rows with future dates beyond the threshold
    future_dates_mask = df['Review_Date'] > FUTURE_DATE_THRESHOLD
    num_future_dates = future_dates_mask.sum()
    if num_future_dates > 0:
        print(f"\nWarning: Found {num_future_dates} reviews with dates beyond {FUTURE_DATE_THRESHOLD}.")
        print("These might be data anomalies (e.g., 'A Google user' reviews with placeholder dates).")
        # Option 1: Filter them out for time-based analysis
        df = df[~future_dates_mask].copy() # Use .copy() to avoid SettingWithCopyWarning
        print(f"Filtered out {num_future_dates} rows with future dates. New row count: {len(df)}")
        # Option 2: Keep them but be mindful during time-series plots
        # Option 3: Analyze them separately if they are numerous or have unique characteristics

# Drop rows where Review_Date became NaT after coercion (if any)
df.dropna(subset=['Review_Date'], inplace=True)
print(f"Rows after handling date issues: {len(df)}")


# --- 1. Rating Distribution ---
print("\n--- 1. Rating Distribution ---")
plt.figure(figsize=(8, 6))
sns.countplot(x='Rating', data=df, palette='viridis', order=df['Rating'].value_counts().index)
plt.title('Distribution of GroMo Partner Ratings')
plt.xlabel('Rating (Stars)')
plt.ylabel('Number of Reviews')
plt.grid(axis='y', linestyle='--')
plt.savefig('rating_distribution.png')
plt.show()

average_rating = df['Rating'].mean()
print(f"Average Rating: {average_rating:.2f} stars")

# --- 2. Review Volume Over Time ---
print("\n--- 2. Review Volume Over Time ---")
df_time = df.set_index('Review_Date')
reviews_per_month = df_time['Rating'].resample('M').count() # 'M' for month-end frequency

if not reviews_per_month.empty:
    plt.figure(figsize=(12, 6))
    reviews_per_month.plot(kind='line', marker='o')
    plt.title('Number of GroMo Partner Reviews Over Time (Monthly)')
    plt.xlabel('Month')
    plt.ylabel('Number of Reviews')
    plt.grid(True)
    plt.savefig('reviews_over_time.png')
    plt.show()
else:
    print("Not enough data points for monthly review volume plot after date filtering.")

# --- 3. Average Rating Over Time ---
print("\n--- 3. Average Rating Over Time ---")
average_rating_per_month = df_time['Rating'].resample('M').mean()

if not average_rating_per_month.empty:
    plt.figure(figsize=(12, 6))
    average_rating_per_month.plot(kind='line', marker='o', color='green')
    plt.title('Average GroMo Partner Rating Over Time (Monthly)')
    plt.xlabel('Month')
    plt.ylabel('Average Rating')
    plt.ylim(1, 5) # Ratings are between 1 and 5
    plt.grid(True)
    plt.axhline(average_rating, color='red', linestyle='--', label=f'Overall Avg ({average_rating:.2f})')
    plt.legend()
    plt.savefig('average_rating_over_time.png')
    plt.show()
else:
    print("Not enough data points for monthly average rating plot after date filtering.")


# --- 4. Text Preprocessing for N-grams and Topic Modeling ---
print("\n--- 4. Text Preprocessing ---")
stop_words = set(stopwords.words('english'))
# Add custom stop words relevant to app reviews or GroMo if needed
custom_stopwords = ['app', 'gromo', 'application', 'please', 'also', 'get', 'even', 'would', 'could', 'make', 'really', 'good', 'great', 'nice', 'best', 'awesome', 'worst', 'bad']
stop_words.update(custom_stopwords)

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text)     # Remove numbers
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2] # Lemmatize and remove short words
    return " ".join(tokens)

# Ensure 'Review_Message' column exists and apply preprocessing
if 'Review_Message' in df.columns:
    print("Preprocessing review messages...")
    df['Processed_Message'] = df['Review_Message'].apply(preprocess_text)
    print("Text preprocessing complete.")
    print(df[['Review_Message', 'Processed_Message']].head())
else:
    print("Error: 'Review_Message' column not found. Cannot perform text analysis.")
    # exit() # Or handle appropriately

# --- 5. Most Common Words/Phrases (N-grams) ---
def plot_top_ngrams(corpus, title, ngram_range=(1,1), top_n=20, filename='top_ngrams.png'):
    if not corpus.empty:
        try:
            vec = CountVectorizer(ngram_range=ngram_range, stop_words=list(stop_words)).fit(corpus)
            bag_of_words = vec.transform(corpus)
            sum_words = bag_of_words.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
            
            top_df = pd.DataFrame(words_freq[:top_n], columns=['Ngram', 'Frequency'])
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Frequency', y='Ngram', data=top_df, palette='mako')
            plt.title(title)
            plt.tight_layout()
            plt.savefig(filename)
            plt.show()
        except ValueError as e:
            print(f"Could not generate n-grams for '{title}': {e}. Corpus might be empty after preprocessing.")
    else:
        print(f"Corpus for '{title}' is empty. Skipping n-gram plot.")

if 'Processed_Message' in df.columns:
    print("\n--- 5. Most Common Words/Phrases (N-grams) ---")
    # For Negative Reviews (1-2 stars)
    negative_reviews_text = df[df['Rating'] <= 2]['Processed_Message'].dropna()
    plot_top_ngrams(negative_reviews_text, 'Top Unigrams in Negative Reviews (1-2 Stars)', ngram_range=(1,1), filename='top_unigrams_negative.png')
    plot_top_ngrams(negative_reviews_text, 'Top Bigrams in Negative Reviews (1-2 Stars)', ngram_range=(2,2), filename='top_bigrams_negative.png')

    # For Positive Reviews (4-5 stars)
    positive_reviews_text = df[df['Rating'] >= 4]['Processed_Message'].dropna()
    plot_top_ngrams(positive_reviews_text, 'Top Unigrams in Positive Reviews (4-5 Stars)', ngram_range=(1,1), filename='top_unigrams_positive.png')

    # Word Cloud for Negative Reviews
    if not negative_reviews_text.empty:
        all_negative_text = " ".join(review for review in negative_reviews_text)
        if all_negative_text.strip(): # Check if string is not empty
            wordcloud = WordCloud(stopwords=stop_words, background_color="white", width=800, height=400).generate(all_negative_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title('Word Cloud for Negative Reviews (1-2 Stars)')
            plt.savefig('wordcloud_negative_reviews.png')
            plt.show()
        else:
            print("No text available for negative review word cloud after processing.")
    else:
        print("No negative reviews found for word cloud.")
else:
    print("Skipping N-gram analysis as 'Processed_Message' is not available.")


# --- 6. Sentiment Analysis (using VADER for better nuance) ---
print("\n--- 6. Sentiment Analysis (VADER) ---")
if 'Review_Message' in df.columns:
    analyzer = SentimentIntensityAnalyzer()
    df['VADER_Sentiment_Compound'] = df['Review_Message'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
    
    def categorize_sentiment(compound_score):
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
            
    df['VADER_Sentiment_Label'] = df['VADER_Sentiment_Compound'].apply(categorize_sentiment)

    print(df[['Rating', 'Review_Message', 'VADER_Sentiment_Compound', 'VADER_Sentiment_Label']].head())

    plt.figure(figsize=(8, 6))
    sns.countplot(x='VADER_Sentiment_Label', data=df, palette='coolwarm', order=['Positive', 'Neutral', 'Negative'])
    plt.title('Sentiment Distribution of Reviews (VADER)')
    plt.xlabel('Sentiment Label')
    plt.ylabel('Number of Reviews')
    plt.savefig('vader_sentiment_distribution.png')
    plt.show()

    # Compare VADER sentiment with Star Rating
    plt.figure(figsize=(10, 7))
    sns.boxplot(x='Rating', y='VADER_Sentiment_Compound', data=df, palette='viridis')
    plt.title('VADER Sentiment Compound Score vs. Star Rating')
    plt.xlabel('Star Rating')
    plt.ylabel('VADER Compound Score (-1 to 1)')
    plt.savefig('vader_vs_star_rating.png')
    plt.show()
else:
    print("Skipping VADER sentiment analysis as 'Review_Message' is not available.")


# --- 7. Developer Engagement Analysis ---
print("\n--- 7. Developer Engagement Analysis ---")
if 'Has_Developer_Reply' in df.columns:
    reply_counts = df['Has_Developer_Reply'].value_counts(normalize=True) * 100
    print("Percentage of Reviews with Developer Reply:")
    print(reply_counts)

    plt.figure(figsize=(7, 5))
    sns.barplot(x=reply_counts.index, y=reply_counts.values, palette=['lightcoral', 'lightgreen'])
    plt.title('Developer Reply Rate')
    plt.xlabel('Has Developer Reply?')
    plt.ylabel('Percentage of Reviews')
    plt.xticks([0, 1], ['No Reply', 'Has Reply'])
    plt.savefig('developer_reply_rate.png')
    plt.show()

    # Reply rate by star rating
    reply_rate_by_rating = df.groupby('Rating')['Has_Developer_Reply'].mean() * 100
    plt.figure(figsize=(10, 6))
    reply_rate_by_rating.plot(kind='bar', color='skyblue')
    plt.title('Developer Reply Rate by Star Rating')
    plt.xlabel('Star Rating')
    plt.ylabel('Percentage of Reviews with Reply (%)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--')
    plt.savefig('developer_reply_rate_by_rating.png')
    plt.show()
else:
    print("Skipping Developer Engagement analysis as 'Has_Developer_Reply' is not available.")


# --- Analysis of "How GroMo Improved With Time and In What Aspects" ---
# This is more complex and requires combining time-series analysis with topic modeling or keyword trends.
# For now, we can look at topic prevalence over time IF we had topic modeling results.
# Let's placeholder this for now. A full Topic Modeling (LDA) implementation is more involved.

print("\n--- Further Analysis Ideas (How GroMo Improved) ---")
print("1. Topic Modeling (e.g., LDA) on 'Processed_Message'.")
print("2. Track prevalence of negative topics (e.g., 'payment delay', 'app crash') over time.")
print("3. Track prevalence of positive topics (e.g., 'easy earning', 'good support') over time.")
print("4. Correlate changes in topic prevalence with average rating changes or app update versions (if available and reliable).")

# --- Example: Simple Keyword Trend Over Time (Illustrative) ---
# This is a basic example. Proper topic modeling would be better.
if 'Processed_Message' in df.columns:
    df_time['Processed_Message'] = df['Processed_Message'] # Add processed messages to df_time

    def check_keyword(text, keyword):
        return keyword in text if pd.notna(text) else False

    # Example keywords for potential problems or improvements
    keywords_of_interest = {
        'payment': ['payment', 'payout', 'withdrawal', 'paid'],
        'support': ['support', 'customer care', 'help', 'helpline', 'query', 'issue resolve'],
        'crash': ['crash', 'bug', 'slow', 'hang', 'lag', 'error'],
        'commission': ['commission', 'earning', 'income', 'incentive'],
        'easy': ['easy', 'simple', 'user friendly', 'smooth']
    }

    plt.figure(figsize=(15, 10))
    plot_index = 1
    for category, kws in keywords_of_interest.items():
        df_time[f'mentions_{category}'] = df_time['Processed_Message'].apply(lambda x: any(kw in str(x) for kw in kws))
        keyword_trend = df_time[f'mentions_{category}'].resample('M').mean() * 100 # Percentage of reviews mentioning

        if not keyword_trend.empty and plot_index <= 4 : # Limit to 4 plots for this figure
            plt.subplot(2, 2, plot_index)
            keyword_trend.plot(kind='line', marker='.')
            plt.title(f'Monthly Trend: Reviews Mentioning "{category.capitalize()}" (%)')
            plt.xlabel('Month')
            plt.ylabel('% of Reviews')
            plt.grid(True)
            plot_index += 1
    
    if plot_index > 1:
        plt.tight_layout()
        plt.savefig('keyword_trends_over_time.png')
        plt.show()
    else:
        print("Not enough data or no keywords found for trend plotting.")
else:
    print("Skipping keyword trend analysis as 'Processed_Message' is not available.")

print("\n--- Analysis Complete ---")
print("Generated plots have been saved as PNG files in the script's directory.")