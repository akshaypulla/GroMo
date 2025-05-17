from google_play_scraper import Sort, reviews_all, app
import pandas as pd

# 1. Define the App ID for GroMo
app_id = 'com.gromo.partner' # GroMo's App ID

# Optional: Get app details to verify
try:
    app_details = app(app_id)
    print(f"Fetching reviews for App: {app_details['title']}")
    print(f"Total Ratings (text reviews count): {app_details['ratings']}")
    print("-" * 30)
except Exception as e:
    print(f"Could not fetch app details: {e}")
    exit()

# 2. Scrape all reviews
print(f"Fetching ALL reviews for {app_id}...")
print("This might take a while depending on the total number of reviews...")
try:
    all_reviews_data = reviews_all(
        app_id,
        sleep_milliseconds=0,  # Default is 0, can be increased if facing throttling
        lang='en',             # Language of reviews (e.g., 'en' for English)
        country='in',          # Country (e.g., 'in' for India, 'us' for US)
        sort=Sort.NEWEST,      # Sort by newest. Other options: Sort.MOST_RELEVANT, Sort.RATING
        # filter_score_with=None # To get all ratings. Or e.g., 5 for only 5-star reviews.
    )
except Exception as e:
    print(f"Error fetching reviews: {e}")
    all_reviews_data = []

if not all_reviews_data:
    print("No reviews found or an error occurred during fetching.")
    exit()

print(f"Successfully fetched {len(all_reviews_data)} reviews.")

# 3. Convert raw review data to a Pandas DataFrame
df_raw_reviews = pd.DataFrame(all_reviews_data)

# 4. Select and rename columns for the desired output
# Mapping from library fields to your desired fields:
# - Rating: 'score'
# - Date of review: 'at'
# - Review message: 'content'
# - How many people found it helpful: 'thumbsUpCount'
# - User name: 'userName'
# - Did he get reply: Check if 'replyContent' is not None/NaN
# - Date of reply: 'repliedAt'
# - Message of reply: 'replyContent'

df_detailed_reviews = pd.DataFrame()

df_detailed_reviews['User_Name'] = df_raw_reviews['userName']
df_detailed_reviews['Review_Date'] = pd.to_datetime(df_raw_reviews['at']) # Convert to datetime objects
df_detailed_reviews['Rating'] = df_raw_reviews['score']
df_detailed_reviews['Review_Message'] = df_raw_reviews['content']
df_detailed_reviews['Helpful_Count'] = df_raw_reviews['thumbsUpCount']

# Developer Reply Information
df_detailed_reviews['Has_Developer_Reply'] = df_raw_reviews['replyContent'].notna()
df_detailed_reviews['Developer_Reply_Message'] = df_raw_reviews['replyContent'].fillna('') # Fill NaN with empty string
df_detailed_reviews['Developer_Reply_Date'] = pd.to_datetime(df_raw_reviews['repliedAt']) # NaT if no reply

# Optional: Include review ID and app version at the time of review
# df_detailed_reviews['Review_ID'] = df_raw_reviews['reviewId']
# df_detailed_reviews['App_Version_At_Review'] = df_raw_reviews['reviewCreatedVersion']


# 5. Display the first few detailed reviews
print("\nFirst 5 detailed reviews:")
print(df_detailed_reviews.head())

# 6. Save the detailed reviews to a CSV file
csv_filename = './gromo_play_store_reviews_detailed.csv'
try:
    df_detailed_reviews.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\nAll detailed reviews saved to {csv_filename}")
except Exception as e:
    print(f"\nError saving to CSV: {e}")
    print("Displaying data in console instead (first 100 rows):")
    print(df_detailed_reviews.head(100).to_string())


# To see all available columns from the raw data:
# print("\nAll available columns from the scraper:")
# print(df_raw_reviews.columns)
# Index(['reviewId', 'userName', 'userImage', 'content', 'score', 'thumbsUpCount',
#        'reviewCreatedVersion', 'at', 'replyContent', 'repliedAt', 'appVersion'],
#       dtype='object')