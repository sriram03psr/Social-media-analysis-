import tweepy
import pandas as pd
#replace
consumer_key = "YOUR_CONSUMER_KEY"
consumer_secret = "YOUR_CONSUMER_SECRET"
access_token = "YOUR_ACCESS_TOKEN"
access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"
keyword = "python"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)
desired_tweet_count = 100
tweets_data = []
for tweet in tweepy.Cursor(api.search, q=keyword, lang="en").items(desired_tweet_count):
    tweets_data.append({
        'Username': tweet.user.screen_name,
        'Tweet': tweet.text,
        'Date': tweet.created_at,
        'Retweets': tweet.retweet_count,
        'Favorites': tweet.favorite_count
    })
tweet = pd.DataFrame(tweets_data)
print(tweets_df)
stopwords = set(STOPWORDS) 
stopwords.add('will')
import re
import seaborn as sns
sns.set()
plt.style.use('seaborn-whitegrid')
def WordCloudPlotter(dfColumn):
    colData = df[dfColumn]
    textCloud = ''
    for mem in colData:
        textCloud = textCloud + str(mem)
    wordcloud = WordCloud(width = 800, height = 800,background_color ='white', 
                          stopwords = stopwords,  min_font_size = 10).generate(textCloud)
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.style.use('seaborn-whitegrid')
    plt.imshow(wordcloud) 
    plt.rcParams.update({'font.size': 25})
    plt.axis("off") 
    plt.title('trending: ' + str(dfColumn))
    plt.tight_layout(pad = 0) 
    plt.show()
WordCloudPlotter('tweet')
