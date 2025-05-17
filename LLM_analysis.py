import pandas as pd
import requests
import json
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3" # Or your preferred model
CSV_FILE_PATH = 'gromo_play_store_reviews_ALL_fetched.csv' # Your full data file
OUTPUT_CSV_FILE_PATH = 'gromo_reviews_monthly_analysis_ollama.csv'
FILTER_FUTURE_DATES = True
FUTURE_DATE_THRESHOLD = pd.Timestamp.now() + pd.Timedelta(days=1)

# --- Load Data and Prepare Time Column ---
print(f"Loading data from {CSV_FILE_PATH}...")
try:
    df_full = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {CSV_FILE_PATH}.")
    exit()

print(f"Original data loaded: {len(df_full)} rows.")

# Convert 'Review_Date' to datetime objects, coercing errors
df_full['Review_Date'] = pd.to_datetime(df_full['Review_Date'], errors='coerce')

# Handle potential 'future dates'
if FILTER_FUTURE_DATES:
    future_dates_mask = df_full['Review_Date'] > FUTURE_DATE_THRESHOLD
    num_future_dates = future_dates_mask.sum()
    if num_future_dates > 0:
        print(f"Warning: Found {num_future_dates} reviews with dates beyond {FUTURE_DATE_THRESHOLD}. Filtering them out for time-based analysis.")
        df_full = df_full[~future_dates_mask].copy()

# Drop rows where Review_Date became NaT or is missing
df_full.dropna(subset=['Review_Date'], inplace=True)
df_full['YearMonth'] = df_full['Review_Date'].dt.to_period('M').astype(str) # e.g., '2023-01'

print(f"Data after date processing: {len(df_full)} rows.")

# Filter for reviews with developer replies for the detailed reply analysis part
df_to_analyze = df_full[
    df_full['Developer_Reply_Message'].notna() & \
    (df_full['Developer_Reply_Message'].str.strip() != '') & \
    df_full['Review_Message'].notna() & \
    (df_full['Review_Message'].str.strip() != '')
].copy()

print(f"Found {len(df_to_analyze)} reviews with developer replies to analyze with LLM.")

# --- Ollama API Interaction Function (same as before) ---
def query_ollama(prompt_text, system_message=""):
    # ... (exact same function as in the previous script) ...
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt_text,
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
        response.raise_for_status()
        full_response_text = response.text
        final_json_response = None
        for line in full_response_text.strip().split('\n'):
            try:
                final_json_response = json.loads(line)
            except json.JSONDecodeError:
                continue
        if final_json_response and "response" in final_json_response:
            return final_json_response["response"].strip()
        else:
            print(f"Warning: 'response' field not found. Full: {full_response_text}")
            return "Error: Could not parse Ollama response."
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return f"Error: API call failed - {e}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"Error: Unexpected - {e}"


# --- Prompts (same as before) ---
SYSTEM_MSG_PAIN_POINT_ID = "You are an expert analyst..." # (As defined before)
USER_PROMPT_PAIN_POINT_ID_TEMPLATE = """Partner's Review Text: ...""" # (As defined before)

SYSTEM_MSG_REPLY_ANALYSIS = "You are an expert analyst evaluating customer support..." # (As defined before)
USER_PROMPT_REPLY_ANALYSIS_TEMPLATE = """Partner's Review Text: ...""" # (As defined before)


# --- Process Reviews (apply to df_to_analyze) ---
analysis_results = []
# sample_df_to_analyze = df_to_analyze.head(10) # For testing
sample_df_to_analyze = df_to_analyze # For full run

for index, row in sample_df_to_analyze.iterrows():
    print(f"\nProcessing review {index + 1}/{len(sample_df_to_analyze)} (Original Index: {row.name})...")
    review_text = str(row['Review_Message'])
    developer_reply = str(row['Developer_Reply_Message'])
    year_month = str(row['YearMonth'])
    rating = row['Rating']

    # LLM Interaction A: Identify Pain Points
    prompt_a = USER_PROMPT_PAIN_POINT_ID_TEMPLATE.format(review_text=review_text)
    print("  Querying Ollama for pain points...")
    identified_pain_points = query_ollama(prompt_a, system_message=SYSTEM_MSG_PAIN_POINT_ID)
    time.sleep(0.5) # Shorter delay, adjust based on your Ollama setup

    # LLM Interaction B: Analyze Developer's Reply
    prompt_b = USER_PROMPT_REPLY_ANALYSIS_TEMPLATE.format(
        review_text=review_text,
        identified_pain_points=identified_pain_points,
        developer_reply=developer_reply
    )
    print("  Querying Ollama for reply analysis...")
    reply_analysis_text = query_ollama(prompt_b, system_message=SYSTEM_MSG_REPLY_ANALYSIS)
    time.sleep(0.5)

    analysis_results.append({
        'Original_Index': row.name, # Original index from df_full
        'YearMonth': year_month,
        'Rating': rating,
        'Review_Message': review_text,
        'Developer_Reply_Message': developer_reply,
        'LLM_Identified_Pain_Points': identified_pain_points,
        'LLM_Reply_Analysis_Raw': reply_analysis_text
    })
    print(f"  Completed processing for Original Index: {row.name}.")


# --- Create DataFrame from results ---
df_llm_analysis = pd.DataFrame(analysis_results)

# --- Parse the structured reply analysis (same as before) ---
def parse_reply_analysis(raw_text):
    # ... (exact same function as in the previous script) ...
    parsed = {
        'Relevance_to_Pain_Points': 'Error parsing',
        'Nature_of_Response': 'Error parsing',
        'Problem_Resolution_Indication': 'Error parsing',
        'Tone_of_Response': 'Error parsing'
    }
    if pd.isna(raw_text) or isinstance(raw_text, str) and "Error:" in raw_text:
        return parsed
    if not isinstance(raw_text, str): # Ensure it's a string before splitting
        return parsed

    lines = raw_text.strip().split('\n')
    for line in lines:
        if "1. Relevance to Pain Points:" in line: # More robust check
            parsed['Relevance_to_Pain_Points'] = line.split(":", 1)[1].strip() if ":" in line else line.strip()
        elif "2. Nature of Response:" in line:
            parsed['Nature_of_Response'] = line.split(":", 1)[1].strip() if ":" in line else line.strip()
        elif "3. Problem Resolution Indication:" in line:
            parsed['Problem_Resolution_Indication'] = line.split(":", 1)[1].strip() if ":" in line else line.strip()
        elif "4. Tone of Response:" in line:
            parsed['Tone_of_Response'] = line.split(":", 1)[1].strip() if ":" in line else line.strip()
    return parsed


if not df_llm_analysis.empty:
    parsed_data = df_llm_analysis['LLM_Reply_Analysis_Raw'].apply(lambda x: pd.Series(parse_reply_analysis(x)))
    df_llm_analysis = pd.concat([df_llm_analysis.drop(columns=['LLM_Reply_Analysis_Raw']), parsed_data], axis=1)

    print("\n--- Sample of LLM Analysis Results (with YearMonth) ---")
    print(df_llm_analysis[['YearMonth', 'Rating', 'LLM_Identified_Pain_Points', 'Relevance_to_Pain_Points']].head())
    df_llm_analysis.to_csv(OUTPUT_CSV_FILE_PATH, index=False, encoding='utf-8-sig')
    print(f"\nLLM analysis with monthly data saved to {OUTPUT_CSV_FILE_PATH}")
else:
    print("\nNo reviews with replies were processed by LLM. Skipping detailed analysis and saving.")


# --- Time-Series Aggregation and Visualization ---
if not df_llm_analysis.empty:
    print("\n--- Monthly Trend Analysis ---")

    # --- Helper function to normalize and count multi-value LLM outputs ---
    def get_monthly_item_counts(df, column_name, item_separator=','):
        monthly_counts = {}
        # Ensure YearMonth is sorted for chronological plotting
        sorted_year_months = sorted(df['YearMonth'].unique())

        for ym in sorted_year_months:
            month_df = df[df['YearMonth'] == ym]
            item_counter = Counter()
            if column_name in month_df.columns:
                for item_list_str in month_df[column_name].dropna():
                    if pd.notna(item_list_str) and isinstance(item_list_str, str):
                         # Handle cases like "1. Pain point one 2. Pain point two" or "Pain point one, Pain point two"
                        items = []
                        # First, try splitting by common list indicators like numbered lists or newlines
                        potential_items = re.split(r'\n\d*\.\s*|\n-\s*|\n\*\s*', item_list_str)
                        for pi in potential_items:
                            pi = pi.strip()
                            if not pi: continue
                            # If still looks like a single line with separators, split by separator
                            if item_separator in pi and len(potential_items) == 1:
                                items.extend([i.strip().lower() for i in pi.split(item_separator) if i.strip()])
                            elif pi.lower() not in ["no specific pain points mentioned", "n/a", "error parsing", "", "none"]:
                                items.append(pi.lower()) # Add cleaned item
                        
                        # Remove generic non-pain points if they slipped through
                        items = [item for item in items if item not in ["no specific pain points mentioned", "n/a", "error parsing", "", "none"]]
                        item_counter.update(items)
            monthly_counts[ym] = item_counter
        return monthly_counts

    # --- 1. Pain Point Trends ---
    # Normalizing LLM_Identified_Pain_Points is tricky due to free text.
    # For a robust version, you'd need:
    #   a) Stricter LLM output format for pain points (e.g., only from a predefined list).
    #   b) A second LLM call to categorize free-text pain points into predefined buckets.
    #   c) Manual categorization or keyword mapping for common free-text pain points.

    # Let's try a simple approach assuming some consistency or using the raw strings for now
    # This will be very noisy without proper normalization.
    print("\nAnalyzing Pain Point Trends (this may be noisy without normalization)...")
    monthly_pain_points = get_monthly_item_counts(df_llm_analysis, 'LLM_Identified_Pain_Points', item_separator='\n')

    # Create a DataFrame for plotting (example for top N overall pain points)
    overall_pain_point_counter = Counter()
    for ym_counts in monthly_pain_points.values():
        overall_pain_point_counter.update(ym_counts)
    
    top_n_pain_points = [item for item, count in overall_pain_point_counter.most_common(5) if item.strip()] # Top 5 non-empty

    if top_n_pain_points:
        pain_point_trends_df = pd.DataFrame(index=sorted(monthly_pain_points.keys()))
        for pp in top_n_pain_points:
            pain_point_trends_df[pp] = [monthly_pain_points[ym].get(pp, 0) for ym in pain_point_trends_df.index]

        plt.figure(figsize=(15, 7))
        for pp in top_n_pain_points:
            plt.plot(pain_point_trends_df.index, pain_point_trends_df[pp], marker='o', label=pp)
        plt.title('Monthly Trend of Top Identified Pain Points (Count)')
        plt.xlabel('Year-Month')
        plt.ylabel('Number of Mentions')
        plt.xticks(rotation=45)
        plt.legend(title='Pain Points', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('monthly_pain_point_trends.png')
        plt.show()
    else:
        print("No consistent pain points found to plot trends, or LLM output needs more normalization.")

    # --- 2. Developer Response Effectiveness Trends ---
    response_metrics_to_track = ['Relevance_to_Pain_Points', 'Nature_of_Response', 'Problem_Resolution_Indication', 'Tone_of_Response']
    for metric_col in response_metrics_to_track:
        if metric_col in df_llm_analysis.columns:
            # Get value counts for each month
            # For simplicity, let's track the proportion of a 'positive' outcome vs others
            # Example: For 'Relevance', track 'Fully Addressed'
            # This needs careful definition of 'positive' outcomes for each metric
            
            # More general: plot distribution of categories over time (stacked bar or line)
            monthly_metric_dist = df_llm_analysis.groupby('YearMonth')[metric_col].value_counts(normalize=True).mul(100).unstack(fill_value=0)
            
            if not monthly_metric_dist.empty:
                # Select top categories to avoid clutter
                top_categories = monthly_metric_dist.sum().nlargest(5).index 
                monthly_metric_dist_top = monthly_metric_dist[top_categories]

                monthly_metric_dist_top.plot(kind='line', figsize=(15, 7), marker='o')
                plt.title(f'Monthly Trend of Developer Response: {metric_col} (%)')
                plt.xlabel('Year-Month')
                plt.ylabel('Percentage of Reviews')
                plt.xticks(rotation=45)
                plt.legend(title=metric_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'monthly_trend_{metric_col.lower()}.png')
                plt.show()
            else:
                print(f"Not enough data to plot trends for {metric_col}.")
        else:
            print(f"Metric column '{metric_col}' not found in analysis results.")
else:
    print("LLM analysis DataFrame is empty. Cannot perform monthly trend analysis.")

print("\nMonthly trend analysis attempt complete. Review plots and CSV.")