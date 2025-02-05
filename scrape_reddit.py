import  praw
import  pandas as pd
import  re
import  os
from    datetime import datetime, timedelta, timezone
from    nltk.sentiment.vader import SentimentIntensityAnalyzer

# Reddit API credentials
client_id       = '8r9Awiioo3VRvmK8ACpZxg'
client_secret   = 'RRRDlRXRMBZhKsvKXwktNDHyTDvY6Q'
user_agent      = 'Stockmarket_Analysis by SpecialistIntrepid21'

# Initialize Reddit instance
reddit = praw.Reddit(client_id      = client_id,
                     client_secret  = client_secret,
                     user_agent     = user_agent)

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Subreddits to scrape
subreddits = [
    'wallstreetbets', 'investing', 'Stocks', 'StockMarket', 
    'WallStreetbetsELITE', 'Daytrading', 'Bogleheads'
]

# Set of common abbreviations to exclude from ticker consideration
excluded_abbreviations = {
    'MAGA', 'TLDR', 'IDK', 'WTF', 'BEARS', 'BEAR', 'BULL', 'BULLS', 'LOL', 'AI',
    'USD', 'USA', 'FOMO', 'PUTS', 'CALLS', 'BTC', 'SELL', 'TRUMP', 'LLC', 'IRA',
    'AM', 'CAD', 'US', 'EWW', 'KEEP', 'YOU', 'RULE', 'RULES', 'NEWS', 'GPU',
    'DOJ', 'CEO', 'DJT', 'EASY', 'SELF', 'GREG', 'EU', 'AFAIK', 'FYI', 'TRUE',
    'MONEY', 'IF', 'AND', 'OR', 'WHY', 'WHEN', 'THEN', 'THE', 'CAN', 'ATM',
    'GDP', 'NEWS', 'FUCK', 'SHIT', 'DAMN', 'HELL', 'PUSSY', 'COCK', 'DICK'
}

# Dictionary mapping company names to their stock tickers
company_name_to_ticker = {
    'apple': 'AAPL',
    'microsoft': 'MSFT',
    'amazon': 'AMZN',
    'nvidia': 'NVDA',
    'tesla': 'TSLA',
    'meta': 'META',
    'netflix': 'NFLX',
    'intel': 'INTC',
    'amd': 'AMD',
    'boeing': 'BA',
    'disney': 'DIS',
    'nike': 'NKE',
    'starbucks': 'SBUX',
    'walmart': 'WMT',
    'visa': 'V',
    'oracle': 'ORCL',
    'adobe': 'ADBE',
    'paypal': 'PYPL',
    'qualcomm': 'QCOM',
    'uber': 'UBER',
    'lyft': 'LYFT',
    'zoom': 'ZM',
    'shopify': 'SHOP',
    'snap': 'SNAP',
    'twitter': 'TWTR',
    'ford': 'F',
    'sony': 'SONY',
    'toyota': 'TM',
    'honda': 'HMC',
    'hyundai': 'HYMTF',
    'nintendo': 'NTDOY',
    'samsung': 'SSNLF',
    'panasonic': 'PCRFY',
    'lg': 'LGEAF'
}

# Date range for the last 30 days
end_date    = datetime.now(timezone.utc)
start_date  = end_date - timedelta(days = 30)

# Regex pattern to detect stock tickers (uppercase letters, 2-5 characters)
ticker_pattern = re.compile(r'\b[A-Z]{2,5}\b')

# Function to scrape posts from the last 30 days
def scrape_recent_posts(subreddit_name, limit=1000):
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []

    for submission in subreddit.new(limit=limit):
        post_time = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
        if start_date <= post_time <= end_date:
            content = submission.title + ' ' + submission.selftext
            content = re.sub(r'http\S+|www\S+', '', content)

            # Normalize content to lowercase for matching
            normalized_content = content.lower()

            # Replace company names with their ticker symbols
            for company_name, ticker in company_name_to_ticker.items():
                normalized_content = normalized_content.replace(company_name, ticker)

            # Extract potential tickers
            tickers_in_post = ticker_pattern.findall(normalized_content)

            # Filter out excluded abbreviations
            tickers_in_post = [ticker for ticker in tickers_in_post if ticker not in excluded_abbreviations]

            if tickers_in_post:
                # Fetch all comments
                submission.comments.replace_more(limit=0)
                all_comments = submission.comments.list()

                # Extract unique commenters
                unique_commenters = {comment.author.name for comment in all_comments if comment.author}

                # Calculate sentiment scores
                sentiment_scores = sia.polarity_scores(content)

                posts_data.append({
                    'subreddit': subreddit_name,
                    'title': submission.title,
                    'selftext': submission.selftext,
                    'tickers': tickers_in_post,
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'unique_commenters_count': len(unique_commenters),
                    'created_utc': post_time,
                    'url': submission.url,
                    'sentiment': sentiment_scores['compound']
                })
    return posts_data



# Scrape data from all subreddits
all_posts = []
for sub in subreddits:
    posts = scrape_recent_posts(sub)
    all_posts.extend(posts)

# Convert to DataFrame
df = pd.DataFrame(all_posts)

# File path for the CSV
file_path = 'reddit_top_trending_stocks_30_days.csv'

# Load existing data if the file exists
if os.path.exists(file_path):
    existing_df = pd.read_csv(file_path, parse_dates=['created_utc'])
    # Combine existing data with new data
    combined_df = pd.concat([existing_df, df]).drop_duplicates(subset = ['url'])
else:
    combined_df = df

# Filter to keep only the last 30 days of data
combined_df = combined_df[combined_df['created_utc'] >= start_date]

# Save the updated DataFrame to CSV
combined_df.to_csv(file_path, index = False)

print(f"Scraped {len(df)} new posts.")
print(f"Dataset now contains {len(combined_df)} posts from the last 30 days.")