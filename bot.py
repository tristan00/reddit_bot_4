
import praw
import sqlite3
import traceback
import logging
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
db_location = r'C:\Users\tdelforge\Documents\project_dbs\reddit\reddit.db'


subreddit_names_to_follow = []

class RedditBot():
    def __init__(self):
        self.topic_models = []

    def set_topic_models(self, topic_models):
        self.topic_models = topic_models

    #for now uses first login in db
    def create_praw_agent(self):
        global user_name
        with sqlite3.connect(db_location) as conn:
            r = conn.execute('select client_id, client_secret, username, password, user_agent from credentials').fetchone()
            client_id, secret, user_name_local, password, user_agent = tuple(r)
            user_name = user_name_local
            reddit_agent = praw.Reddit(client_id = client_id,
                                 client_secret = secret,
                                 username = user_name_local,
                                 password = password,
                                 user_agent = user_agent)
            logger.info('Logged in successfully as {0}'.format(user_name))
        return reddit_agent

    def wipe_db(self):
        with sqlite3.connect(db_location) as conn:
            conn.execute('delete from comments')
            conn.execute('delete from posts')
            conn.execute('delete from subreddits')
            conn.execute('delete from redditors')

    def wipe_preprocessing_db(self):
        with sqlite3.connect(db_location) as conn:
            conn.execute('delete from preprocessed_comments')
            conn.execute('delete from preprocessed_posts')
            conn.commit()


    def build_db(self):
        with sqlite3.connect(db_location) as conn:
            conn.execute('create table if not exists credentials (client_id TEXT UNIQUE, client_secret TEXT, username TEXT, password TEXT, user_agent TEXT)')
            conn.execute('create table if not exists comments (c_id TEXT UNIQUE, p_id TEXT, s_id TEXT, author TEXT, parent_id TEXT, body TEXT, score int, submitted_timestamp TEXT, edited int)')
            conn.execute('create table if not exists posts (p_id TEXT UNIQUE, s_id TEXT, author TEXT, title TEXT, body TEXT, score int, timestamp text, edited int)')
            conn.execute('create table if not exists subreddits (s_id TEXT UNIQUE, display_name TEXT, full_name TEXT, subscribers int, banned int)')
            conn.execute('create table if not exists redditors (redditor_name TEXT UNIQUE, path TEXT)')
            conn.execute('create table if not exists preprocessed_comments (c_id TEXT UNIQUE, input TEXT)')
            conn.execute('create table if not exists preprocessed_posts (p_id TEXT UNIQUE, input TEXT)')
            conn.commit()

    def write_comment_to_db(self, comment, post, conn, commit = False):
        if isinstance(comment, praw.models.MoreComments):
            return
        if comment.author is None:
            return
        adj_parent_id = comment.parent_id.split('_')[1]
        if post.id == adj_parent_id:
            adj_parent_id = None
        try:
            conn.execute('insert into comments values (?, ?, ?, ?, ?, ?, ?, ?, ?)', (comment.id, post.id, post.subreddit_id, comment.author.name, adj_parent_id, comment.body, comment.score, comment.created_utc, int(comment.edited)))
        except sqlite3.IntegrityError:
            conn.execute('update comments set body = ?, score = ?, edited = ? where c_id = ?', (comment.body, comment.score, int(comment.edited), comment.id))

        if commit:
            conn.commit()

    def write_post_to_db(self, post, conn, commit = False):
        if post.author is None:
            return
        try:
            conn.execute('insert into posts values (?, ?, ?, ?, ?, ?, ?, ?)', (post.id, post.subreddit_id, post.author.name, post.title, post.selftext, post.score, post.created_utc, int(post.edited)))
        except sqlite3.IntegrityError:
            conn.execute('update posts set body = ?, score = ?, edited = ? where p_id = ?', (post.selftext, post.score, int(post.edited), post.id))

        if commit:
            conn.commit()

    def write_subreddit_to_db(self, posts, subreddit, conn, commit = False):
        if len(posts) == 0:
            return
        try:
            conn.execute('insert into subreddits values (?, ?, ?, ?, ?)', (posts[0].subreddit_id, subreddit.display_name, subreddit.fullname, subreddit.subscribers, 0))
        except sqlite3.IntegrityError:
            conn.execute('update subreddits set subscribers = ? where s_id = ?', (subreddit.subscribers, posts[0].subreddit_id))
        if commit:
            conn.commit()

    def read_and_store_post_to_db(self, post, conn, write_to_db = True):
        self.write_post_to_db(post, conn)
        post.comments.replace_more()
        for comment in post.comments.list():
            self.write_comment_to_db(comment, post, conn)
            conn.commit()

    #get top, hot and new posts, important to not only get the top posts
    def read_and_store_subreddit_info_to_db(self, subreddit, write_to_db = True):
        logger.info('Starting to read subreddit: {0}'.format(subreddit.display_name))
        start_time = time.time()
        try:
            subreddit.subscribe()
        except:
            traceback.print_exc()

        posts = []
        posts.extend([p for p in subreddit.hot()])
        posts.extend([p for p in subreddit.new()])
        if write_to_db:
            with sqlite3.connect(db_location) as conn:
                self.write_subreddit_to_db(posts, subreddit, conn)
                for post_count, post in enumerate(posts):
                    self.read_and_store_post_to_db(post, conn, write_to_db=write_to_db)
        return posts

    def get_new_posts(self, reddit_agent, num_of_subs = 10):
        logger.info('Getting new data from selected subreddits:')
        if num_of_subs < len(subreddit_names_to_follow):
            subreddits_to_check = random.sample(subreddit_names_to_follow, num_of_subs)
        else:
            subreddits_to_check = subreddit_names_to_follow
        for i in subreddits_to_check:
            try:
                self.read_and_store_subreddit_info_to_db(reddit_agent.subreddit(i))
                self.print_db_size()
            except:
                traceback.print_exc()
        logger.info('New data collected')

    #get a number of past post ids by sampling the full list, updates the data
    def update_stored_posts(self, reddit_agent, num_of_posts=100):
        with sqlite3.connect(db_location) as conn:
            post_ids = conn.execute('select p_id from posts').fetchall()
            try:
                selected_post_ids = random.sample(post_ids, num_of_posts)
            except ValueError:
                #not enough posts
                selected_post_ids = post_ids
            selected_post_ids = [i[0] for i in selected_post_ids]#make p_ids a str
            for p in selected_post_ids:
                p = reddit_agent.submission(id=p)
                self.read_and_store_post_to_db(p, conn)
            conn.commit()
            logger.info('Updated {0} posts:'.format(num_of_posts))

    def print_db_size(self):
        with sqlite3.connect(db_location) as conn:
            num_of_comments = conn.execute('select count(*) from comments').fetchall()[0]
            num_of_posts = conn.execute('select count(*) from posts').fetchall()[0]
            num_of_subreddits = conn.execute('select count(*) from subreddits').fetchall()[0]
            num_of_comments_pre = conn.execute('select count(*) from comments').fetchall()[0]
            num_of_posts_pre = conn.execute('select count(*) from posts').fetchall()[0]
            logger.info('DB size, subreddits: {0}, posts: {1}, comments:{2}'.format(num_of_subreddits[0], num_of_posts[0], num_of_comments[0]))

    def get_comments_for_most_recent_posts(self, num_of_posts = 10):
        with sqlite3.connect(db_location) as conn:
            res = conn.execute('''select a.c_id, a.s_id, c.input, d.input
            from comments a join posts b on a.p_id = b.p_id
            join preprocessed_comments c on a.c_id = c.c_id
            join preprocessed_posts d on a.p_id = d.p_id
            order by b.timestamp desc''').fetchall()
            results = []
            for i in res[0:num_of_posts]:
                results.append({'parent_id':i[0],
                                's_id':i[1],
                                'parent_input':i[2],
                                'post_input':i[3]})
            return results

    def post_reply(self, parent_id, text):
        reddit_agent = self.create_praw_agent()
        parent_comment = reddit_agent.comment(id=parent_id)
        logger.info('posted comment: {0} {1}'.format(parent_id, text))
        parent_comment.reply(text)

    def read_data(self, reddit_agent):
        self.update_stored_posts(reddit_agent)
        self.get_new_posts(reddit_agent)
        return reddit_agent

    def set_up(self):
        self.build_db()
        reddit_agent = self.create_praw_agent()
        return reddit_agent

    def scrape_one_iteration(self):
        reddit_agent = self.set_up()
        self.read_data(reddit_agent)


def get_subreddit_list():
    global subreddit_names_to_follow
    with sqlite3.connect(db_location) as conn:
        res = conn.execute('select distinct display_name from subreddits order by s_id').fetchall()
        subreddit_names_to_follow = [i[0] for i in res]

def main(iterations=100):
    #wipe_db()
    get_subreddit_list()
    bot = RedditBot()
    for i in range(iterations):
        try:
            bot.scrape_one_iteration()
        except:
            traceback.print_exc()
            time.sleep(10)

if __name__ == '__main__':
    main()
    #reddit_agent = set_up()
    #post_reply(reddit_agent, 'dq21ahz', 'yes')
