# reddit_bot_4

reddit bot 1, naive method similar to localized regression, some success on the simpler subreddits
reddit bot 3, dnn to clasifiy comments, would attempt to reuse comments, rank them  by classification post the top one, no success
reddit bot 4, lstm to generate comments, some undetermined regression algorithm to predict success


progress
built comment chains that should be able to be converted to features for training

next step
look into how to lsit features, having n most common words would be huge, may be able to use word to vec
will send in word along side with subreddit and pos