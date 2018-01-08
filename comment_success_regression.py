from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import sqlite3
import pandas as pd
import random
import pickle
import time
import nltk
import datetime
import numpy as np

db_location = r'C:\Users\tdelforge\Documents\project_dbs\reddit\reddit.db'
sorted_subreddit_list = []

pos_tag_map = {'CC':1,'CD':2,'DT':3,'EX':4,'FW':5,'IN':6, 'JJ':7,'JJR':8,'JJS':9, 'LS':10,'MD':11,'NN':12, 'NNS':13, 'NNP':14, 'NNPS': 15,
               'PDT':16, 'POS':17, 'PRP':18, 'PRP$':19,'RB':20,'RBR':21, 'RBS':22, 'RP':23, 'SYM':24, 'TO':25,'UH':26, 'VB':27, 'VBD':28, 'VBG':29, 'VBN':30,
               'VBP':31, 'VBZ':32, 'WDT':33, 'WP':34, 'WP$':35, 'WRB':36}


def report(results, n_top=30):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def get_subreddit_features(s_id):
    global sorted_subreddit_list

    if len(sorted_subreddit_list) == 0:
        with sqlite3.connect(db_location) as conn:
            res = conn.execute('select distinct s_id from subreddits').fetchall()
        sorted_subreddit_list = sorted([i[0] for i in res])
    subreddit_features = [0 for _ in sorted_subreddit_list]
    subreddit_features[sorted_subreddit_list.index(s_id)] = 1
    return subreddit_features


def get_num_of_words(comment_text):
    return len(nltk.tokenize.word_tokenize(comment_text))


def get_num_of_characters(comment_text):
    return len(comment_text)


def get_child_list(df, c_id):
    child_df = df[df['parent_id'] == c_id]
    child_df = child_df.sort_values(by='score', ascending=False)
    if child_df.shape[0] == 0:
        return []
    else:
        return [{'body':child_df['body'].iloc[0], 'score':child_df['score'].iloc[0], 's_id':child_df['s_id'].iloc[0], 'timestamp':child_df['submitted_timestamp'].iloc[0]}] + get_child_list(df, child_df['c_id'].iloc[0])


def get_data():
    with sqlite3.connect(db_location) as conn:
        posts = conn.execute('select p_id, title, timestamp from posts order by score desc').fetchall()

        comments_df = pd.read_sql('select * from comments', conn)
        comment_chains = []
        for count, (p_id, title, timestamp) in enumerate(posts):
            if count > 1000:
                break
            print(count, len(comment_chains), time.time())
            #df = pd.read_sql('select * from comments where p_id = ?', conn, params=(p_id,))
            post_comments_df = comments_df[comments_df['p_id'] == p_id]
            roots = post_comments_df[post_comments_df['parent_id'].isnull()]
            for _, root in roots.iterrows():
                comment_chain = []
                comment_chain.append({'body':title , 'score':None, 's_id':root['s_id'], 'timestamp':timestamp})
                comment_chain.append({'body':root['body'], 'score':root['score'], 's_id':root['s_id'], 'timestamp':root['submitted_timestamp']})
                child_list = get_child_list(post_comments_df, root['c_id'])
                if len(child_list) > 0:
                    comment_chain.extend(child_list)
                    comment_chains.append(comment_chain)
        random.shuffle(comment_chains)

        with open('models/regressor_input.plk', 'wb') as infile:
            pickle.dump(comment_chains, infile)


def get_pos_features(pos_list):
    feature_list = [0 for _ in range(len(pos_tag_map) + 1)]
    for i in pos_list:
        pos_tag = i[1]
        pos_index = pos_tag_map.get(pos_tag, 0)
        feature_list[pos_index] += 1

    return [i/max(1, len(pos_list)) for i in feature_list]


def get_features_for_comment(chain, num):
    title = chain[0]
    previous_comments = chain[1:num]
    tested_comment = chain[num]

    title_timestamp = int(float(title['timestamp']))
    title_datetime = datetime.datetime.fromtimestamp(title_timestamp)
    title_day_of_week = title_datetime.weekday()
    title_month_of_year = title_datetime.month
    title_hour_of_day = title_datetime.hour

    title_weekday_features = [0 for i in range(7)]
    title_weekday_features[title_day_of_week] = 1
    title_month_of_year_features = [0 for i in range(13)]
    title_month_of_year_features[title_month_of_year] = 1
    title_hour_of_day_features = [0 for i in range(24)]
    title_hour_of_day_features[title_hour_of_day] = 1


    previous_comment_timestamp = int(float(chain[num-1]['timestamp']))
    average_time_between_comments = (previous_comment_timestamp - title_timestamp)/(num - 1)
    comment_timestamp = int(float(chain[num]['timestamp']))
    time_since_last_comment = previous_comment_timestamp - title_timestamp

    title_pos = nltk.pos_tag(title['body'])
    previous_comments_pos = get_pos_features(nltk.pos_tag(nltk.tokenize.word_tokenize(' '.join([i['body'] for i in previous_comments]))))
    previous_comment_pos = get_pos_features(nltk.pos_tag(nltk.tokenize.word_tokenize(previous_comments[-1]['body'])))
    tested_comment_pos = get_pos_features(nltk.pos_tag(nltk.tokenize.word_tokenize(tested_comment['body'])))

    subreddit_features = get_subreddit_features(title['s_id'])

    comment_score = tested_comment['score']

    title_word_len = get_num_of_words(title['body'])
    title_average_word_len = get_num_of_characters(title['body'])/title_word_len

    previous_comments_sentences_concatenated = ' '.join([i['body'] for i in previous_comments])
    previous_comments_average_comment_lengths = get_num_of_words(previous_comments_sentences_concatenated)/len(previous_comments)
    previous_comments_average_word_lengths = get_num_of_characters(previous_comments_sentences_concatenated) / max(1, (len(previous_comments) * previous_comments_average_comment_lengths))

    previous_comment_sentences_concatenated = ' '.join([i['body'] for i in previous_comments])
    previous_comment_average_comment_lengths = get_num_of_words(previous_comment_sentences_concatenated)
    previous_comment_average_word_lengths = get_num_of_characters(previous_comment_sentences_concatenated)/max(1, previous_comment_average_comment_lengths)

    x_features = title_weekday_features + title_month_of_year_features + title_hour_of_day_features + \
                 [average_time_between_comments,  time_since_last_comment] + previous_comments_pos + \
                 previous_comment_pos + tested_comment_pos + subreddit_features + \
                [title_average_word_len, previous_comments_average_comment_lengths, previous_comments_average_word_lengths,\
                 previous_comment_average_comment_lengths, previous_comment_average_word_lengths, len(previous_comments)]
    y_features = [comment_score]

    x_features = np.array(x_features)
    y_features = np.array(y_features)

    return [x_features, y_features]


def get_features_from_chain(chain):
    chain_len = len(chain)

    feature_list = []
    for i in range(2, chain_len):
        feature_list.append(get_features_for_comment(chain, i))
    return feature_list


def generate_features(comment_chains):
    feature_list = []
    for count, chain in enumerate(comment_chains):
        feature_list.extend(get_features_from_chain(chain))
        print(count, len(feature_list))
    #x_input = np.vstack([i[0] for i in feature_list])
    #y_input = np.vstack([i[0] for i in feature_list])
    with open('models/regressor_features.plk', 'wb') as infile:
        pickle.dump(feature_list, infile)


def run_model():
    with open('models/regressor_features.plk', 'rb') as infile:
        feature_list = pickle.load(infile)
    x_input = np.vstack([i[0] for i in feature_list])
    y_input = np.vstack([i[1] for i in feature_list])

    min_max_scaler = preprocessing.MinMaxScaler()
    x_input = min_max_scaler.fit_transform(x_input)

    y_input = np.reshape(np.reshape(y_input, (y_input.shape[0],)), (-1, 1))
    print(y_input.shape)
    quant_scaler = preprocessing.QuantileTransformer()
    y_input = quant_scaler.fit_transform(y_input)
    y_input = np.ravel(y_input)

    # clf1 = GradientBoostingRegressor()
    # param_grid1 = {"n_estimators":[100, 500],
    #                 "max_depth": [5, 10],
    #                 "max_features": [None]}
    # grid_search1 = GridSearchCV(clf1, param_grid=param_grid1, verbose=3)
    # grid_search1.fit(x_input, y_input)
    # report(grid_search1.cv_results_)

    # clf2 = RandomForestRegressor()
    # param_grid2 = {"n_estimators":[4, 256],
    #                 "max_depth": [5, 20, 25],
    #                 "max_features": ['sqrt', None]}
    # grid_search2 = GridSearchCV(clf2, param_grid=param_grid2, verbose=3)
    # grid_search2.fit(x_input, y_input)
    # report(grid_search2.cv_results_)

    clf3 = ExtraTreesRegressor()
    param_grid3 = {"n_estimators":[256, 512],
                    "max_depth": [20, 25, 50],
                    "max_features": ['sqrt', None]}
    grid_search3 = GridSearchCV(clf3, param_grid=param_grid3, verbose=3)
    grid_search3.fit(x_input, y_input)
    # print()
    # print()
    # print()
    # print('result 1:')
    # report(grid_search1.cv_results_)
    # print()
    # print('result 2:')
    # report(grid_search2.cv_results_)
    # print()
    # print('result 3:')
    report(grid_search3.cv_results_)


def process_input():
    with open('models/regressor_input.plk', 'rb') as infile:
        comment_chains = pickle.load(infile)
    generate_features(comment_chains)


def main():
    #process_input()
    run_model()


if __name__ == '__main__':
    #get_data()
    main()
