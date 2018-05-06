import numpy as np
import re
import pdb

tweets = ['Once upon a time there were three bears.',
          'Wherever you go, there you are.',
          'Fun Smells',
          'Friendly as a river, fly as a brunette.']

non_words_regex = re.compile('[^a-z ]')

tweets_clean = [non_words_regex.sub('', tweet.lower()) for tweet in tweets]

tweets_as_tuples = [tuple(words.split()) for words in tweets_clean]

words = set()
words_per_tweet = set()


for tweet in tweets_as_tuples:
    for word in tweet:
        words.add(word)
        words_per_tweet.add(len(tweet))

# Initialize word_index with an empty string for zero
word_index = {'': 0}
for word, index in zip(words, range(1, len(words) + 1)):
    word_index[word] = index

num_samples         = len(tweets_as_tuples)
max_words_per_tweet = max(words_per_tweet)
num_words_in_dict   = len(word_index)

# Initialize training data to be encoded as one-hot
training_data = np.zeros((num_samples, max_words_per_tweet, num_words_in_dict), 'int32')

# Add one-hot key
for sample, tweet in zip(training_data, tweets_as_tuples):
    for word, sample_column in zip(tweet, sample):
        row_index = word_index[word]
        sample_column[row_index] = 1.


# As an exercise, generate original tweets_as_tuples from training_data
inverse_word_index = { v: k for k, v in word_index.items() }

reconstructed_tweets = []
for sample in training_data:
    reconstructed_tweet = []
    for sample_column in sample:
        index = sample_column.argmax()
        word = inverse_word_index[index]
        reconstructed_tweet.append(word)
    reconstructed_tweets.append(reconstructed_tweet)




for tweet in reconstructed_tweets:
    print(tweet)


