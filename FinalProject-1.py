# importing all necessary libraries
import praw
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import matplotlib.pyplot as plt
from gensim import corpora, models
from transformers import pipeline

# Downloading NLTK Vader Lexicon (Sentiment Analysis)
nltk.download('vader_lexicon')

# Initializing Reddit API with your credentials
reddit = praw.Reddit(client_id='dhdghdx',
                     client_secret='yfyfjmfu',
                     user_agent='SocialMediaUnrest')   #Replace the credentials

# Defining the subreddits related to your case study
subreddits = ['BlackLivesMatter', 'stopasianhate', 'immigration']

# Initializing SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Using Function for text preprocessing
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'[^\w\s]', '', text) 
    return text

# Initializing variables for aggregated sentiment scores
total_positive = 0
total_negative = 0
total_neutral = 0
total_comments = 0
cleaned_comments = []

# Initializing BERT for sentiment analysis
nlp = pipeline("sentiment-analysis")

for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    
    print(f"Top posts from r/{subreddit_name}:")
    for submission in subreddit.top(limit=25):  
        print(f"Title: {submission.title}")
        print(f"Score: {submission.score}")
        print(f"Comments: {submission.num_comments}")
        print(f"URL: {submission.url}")
        print("\n")

        submission.comments.replace_more(limit=None)
        comments = submission.comments.list()

        print("\nComments and their sentiments:")
        for comment in comments[:10]:  
            comment_text = comment.body
            comment_text = preprocess_text(comment_text)  
            cleaned_comments.append(comment_text)
            
            # Sentiment analysis using NLTK Vader
            sentiment = sia.polarity_scores(comment_text)
            
            # Aggregating sentiment scores
            if sentiment['compound'] >= 0.05:
                total_positive += 1
            elif sentiment['compound'] <= -0.05:
                total_negative += 1
            else:
                total_neutral += 1
            total_comments += 1

# Calculate percentages of sentiments
percentage_positive = (total_positive / total_comments) * 100
percentage_negative = (total_negative / total_comments) * 100
percentage_neutral = (total_neutral / total_comments) * 100

# Print aggregated sentiment results
print(f"Total Comments Analyzed: {total_comments}")
print(f"Percentage of Positive Comments: {percentage_positive:.2f}%")
print(f"Percentage of Negative Comments: {percentage_negative:.2f}%")
print(f"Percentage of Neutral Comments: {percentage_neutral:.2f}%")

# Perform BERT sentiment analysis
bert_sentiments = nlp(cleaned_comments)

# Visualizing sentiments
sentiment_labels = ['Positive', 'Neutral', 'Negative']
sentiment_percentages = [percentage_positive, percentage_neutral, percentage_negative]

plt.bar(sentiment_labels, sentiment_percentages)
plt.xlabel('Sentiment')
plt.ylabel('Percentage')
plt.title('Sentiment Distribution')
plt.show()

# Prepare text for topic modeling
text_data = [comment.split() for comment in cleaned_comments]

# Create a dictionary from the text data
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]

# Perform LDA topic modeling
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary)

