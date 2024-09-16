import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from collections import OrderedDict
import json

nltk.download('stopwords')


class Recommender:

    def __init__(self, json_data=None, user=None):
        self.data = pd.DataFrame(json_data)

        self.user = user
        if user is not None:
            self.user_keywords = user['user_keywords']
            self.user_categories = user['user_categories']
            self.user_view_history = user['user_history']
            self.user_likes = user['liked_articles']
            self.user_dislikes = user['disliked_articles']
        else:
            self.user_keywords = None
            self.user_categories = None
            self.user_view_history = None
            self.user_likes = None
            self.user_dislikes = None

        self.vectors = None


    def load_data(self, json_data):
        self.data = pd.DataFrame(json_data)
        self.delete_missing_values()
        print("Deleted rows with missing values.")
        self.text_preprocessing()
        print("Preprocessed text.")
        

    def load_user(self, user):
        self.user = user
        self.user_keywords = user['user_keywords']
        self.user_categories = user['user_categories']
        self.user_view_history = user['user_history']
        self.user_likes = user['liked_articles']
        self.user_dislikes = user['disliked_articles']




    def delete_missing_values(self):
        # Check for NaN, missing, or NaT values
        missing_rows = self.data[self.data.isnull().any(axis=1)]

        if not missing_rows.empty:
            print("Deleting rows with missing values...")
            self.data.dropna(inplace=True)
            print("Rows with missing values have been deleted.")
        else:
            print("There are no missing values in the .DataFrame.")

    def text_preprocessing(self):

        def _remove_non_ascii(text):
            return ''.join(i for i in text if ord(i) < 128)
        
        def make_lower_case(text):
            return text.lower()
        
        def remove_stop_words(text):
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)
            return text
        
        def remove_html(text):
            html_pattern = re.compile('<.*?>')
            return html_pattern.sub(r'', text)
        
        def remove_punctuation(text):
            tokenizer = RegexpTokenizer(r'\w+')
            text = tokenizer.tokenize(text)
            text = " ".join(text)
            return text
        
        def remove_digits(text):
            return re.sub(r'\d+', '', text)
        
        def decode_unicode_escape(text):
            decoded_text = bytes(text, "utf-8").decode("unicode_escape")
            return decoded_text
        

        print(self.data.columns)

        self.data['title'] = self.data['title'].apply(_remove_non_ascii)
        self.data['title'] = self.data['title'].apply(make_lower_case)
        self.data['title'] = self.data['title'].apply(remove_stop_words)
        self.data['title'] = self.data['title'].apply(remove_html)
        self.data['title'] = self.data['title'].apply(remove_punctuation)
        self.data['title'] = self.data['title'].apply(remove_digits)
        self.data['title'] = self.data['title'].apply(decode_unicode_escape)

        self.data['short_description'] = self.data['short_description'].apply(_remove_non_ascii)
        self.data['short_description'] = self.data['short_description'].apply(make_lower_case)
        self.data['short_description'] = self.data['short_description'].apply(remove_stop_words)
        self.data['short_description'] = self.data['short_description'].apply(remove_html)
        self.data['short_description'] = self.data['short_description'].apply(remove_punctuation)
        self.data['short_description'] = self.data['short_description'].apply(remove_digits)
        self.data['short_description'] = self.data['short_description'].apply(decode_unicode_escape)

        self.data['description'] = self.data['description'].apply(make_lower_case)
        self.data['description'] = self.data['description'].apply(remove_stop_words)
        self.data['description'] = self.data['description'].apply(_remove_non_ascii)
        self.data['description'] = self.data['description'].apply(remove_html)
        self.data['description'] = self.data['description'].apply(remove_punctuation)
        self.data['description'] = self.data['description'].apply(remove_digits)
        self.data['description'] = self.data['description'].apply(decode_unicode_escape)
    
    def load_glove_model(self, path='gloves/converted_vectors300.txt'):
        self.vectors = KeyedVectors.load_word2vec_format(path, binary=False)

    def get_average_word2vec(self, tokens_list, generate_missing=False, k=50):
        if len(tokens_list) < 1:
            return np.zeros(k)
        
        vectorized = []
        for word in tokens_list:
            if word in self.vectors:
                vectorized.append(self.vectors[word])
            elif generate_missing:
                vectorized.append(np.random.rand(k))
        
        if not vectorized:
            return np.zeros(k)
        
        # Only compute the mean if there are non-zero vectors
        if len(vectorized) > 0:
            return np.mean(vectorized, axis=0)
        else:
            return np.zeros(k)

    def recommend_news_based_on_keyword(self, keyword):
        keyword = keyword.split()
        keyword = self.get_average_word2vec(keyword, self.vectors)
        
        # Calculate vectors for title
        # cut down the title into lists of words before applying the get_average_word2vec functio
        self.data['title_vector'] = self.data['title'].apply(lambda x: self.get_average_word2vec(x.split(), self.vectors))
        self.data['short_desc_vector'] = self.data['short_description'].apply(lambda x: self.get_average_word2vec(x.split(), self.vectors))
        self.data['desc_vector'] = self.data['description'].apply(lambda x: self.get_average_word2vec(x.split(), self.vectors))

            # Compute cosine similarity for each title vector
        self.data['title_similarity'] = self.data.apply(lambda row: cosine_similarity([row['title_vector']], [keyword])[0][0], axis=1)
        self.data['short_desc_similarity'] = self.data.apply(lambda row: cosine_similarity([row['short_desc_vector']], [keyword])[0][0], axis=1)
        self.data['desc_similarity'] = self.data.apply(lambda row: cosine_similarity([row['desc_vector']], [keyword])[0][0], axis=1)

        # Combine similarities with weights
        title_weight = 0.7
        short_desc_weight = 0.2
        desc_weight = 0.1

        self.data['combined_similarity'] = (title_weight * self.data['title_similarity'] +
                                                short_desc_weight * self.data['short_desc_similarity'] + 
                                                desc_weight * self.data['desc_similarity'])

            # Sort by title similarity
        self.data = self.data.sort_values(by='title_similarity', ascending=False)

        # Extract the only 50 of id of the sorted data to be used for recommendation
        recommended_ids = self.data['id'][:50]
            
        return recommended_ids

    def recommend_news_based_on_author(self, author, recent=False):
        vectors = self.vectors
        author = author.split()
        author = self.get_average_word2vec(author, vectors)
        
        # Calculate vectors for title
        # cut down the title into lists of words before applying the get_average_word2vec functio
        self.data['author_vector'] = self.data['author'].apply(lambda x: self.get_average_word2vec(x.split(), vectors))

        # Compute cosine similarity for each title vector
        self.data['author_similarity'] = self.data.apply(lambda row: cosine_similarity([row['author_vector']], [author])[0][0], axis=1)

        # Sort by title similarity
        self.data = self.data.sort_values(by='author_similarity', ascending=False)

        # Sort by date_created if recent is True
        if recent:
            self.data = self.data.sort_values(by=['author_similarity', 'date_created'], ascending=[False, False])

        # Extract the only 50 of id of the sorted data to be used for recommendation
        recommended_ids = self.data['id'][:50]

        return recommended_ids

    # def recommend_news_based_on_keyword_and_preferences(self):
    #     user_data = self.user
    #     user_keywords = user_data.get('user_keywords', [])
    #     user_categories = user_data.get('user_categories', [])
    #
    #     if not user_keywords:
    #         return pd.DataFrame()  # Return an empty DataFrame if no user keywords are provided
    #
    #     keyword = ' '.join(user_keywords)
    #     keyword_vector = self.get_average_word2vec(keyword.split(), self.vectors)
    #
    #     # Calculate vectors for title, short description, and description
    #     self.data['title_vector'] = self.data['title'].apply(lambda x: self.get_average_word2vec(x.split(), self.vectors))
    #     self.data['short_desc_vector'] = self.data['short_description'].apply(
    #         lambda x: self.get_average_word2vec(x.split(), self.vectors))
    #     self.data['desc_vector'] = self.data['description'].apply(
    #         lambda x: self.get_average_word2vec(x.split(), self.vectors))
    #
    #     # Compute cosine similarity for each feature
    #     self.data['title_similarity'] = self.data.apply(
    #         lambda row: cosine_similarity([row['title_vector']], [keyword_vector])[0][0], axis=1)
    #     self.data['short_desc_similarity'] = self.data.apply(
    #         lambda row: cosine_similarity([row['short_desc_vector']], [keyword_vector])[0][0], axis=1)
    #     self.data['desc_similarity'] = self.data.apply(
    #         lambda row: cosine_similarity([row['desc_vector']], [keyword_vector])[0][0], axis=1)
    #
    #     # Combine similarities with weights
    #     title_weight = 0.7
    #     short_desc_weight = 0.2
    #     desc_weight = 0.1
    #
    #     self.data['combined_similarity'] = (title_weight * self.data['title_similarity'] +
    #                                         short_desc_weight * self.data['short_desc_similarity'] +
    #                                         desc_weight * self.data['desc_similarity'])
    #
    #     # Sort by combined similarity
    #     self.data = self.data.sort_values(by='combined_similarity', ascending=False)
    #
    #     # Prioritize articles based on user preferences for categories
    #     if user_categories:
    #         category_weights = user_categories
    #         category_max_score = max(category_weights.values(), key=lambda x: x['score'])['score']
    #         for category, weight_info in category_weights.items():
    #             weight = weight_info['score']
    #             self.data.loc[self.data['category'] == category, 'combined_similarity'] *= (weight / category_max_score)
    #
    #     # Sort again by adjusted combined similarity
    #     self.data = self.data.sort_values(by='combined_similarity', ascending=False)
    #
    #     # Remove unnecessary columns
    #     self.data = self.data.drop(columns=['title_vector', 'short_desc_vector', 'desc_vector', 'title_similarity',
    #                                         'short_desc_similarity', 'desc_similarity'])
    #
    #     return self.data


    def recommend_news_based_on_keywords_and_preferences(self):
        data = self.data
        keywords = self.user_keywords
        user_preferences = self.user_categories

        # Vectorize text columns
        data['title_vector'] = data['title'].apply(lambda x: self.get_average_word2vec(x.split(), self.vectors))
        data['short_desc_vector'] = data['short_description'].apply(
            lambda x: self.get_average_word2vec(x.split(), self.vectors))
        data['desc_vector'] = data['description'].apply(lambda x: self.get_average_word2vec(x.split(), self.vectors))

        # Initialize combined similarity column
        data['combined_similarity'] = 0.0

        # Iterate through each keyword and its score
        for keyword, score_info in keywords.items():
            score = score_info['score']
            # Calculate average word2vec representation for the keyword
            keyword_vec = self.get_average_word2vec(keyword.split(), self.vectors)

            # Calculate cosine similarity for each feature with the keyword
            data['title_similarity'] = data.apply(
                lambda row: cosine_similarity([row['title_vector']], [keyword_vec])[0][0], axis=1)
            data['short_desc_similarity'] = data.apply(
                lambda row: cosine_similarity([row['short_desc_vector']], [keyword_vec])[0][0], axis=1)
            data['desc_similarity'] = data.apply(
                lambda row: cosine_similarity([row['desc_vector']], [keyword_vec])[0][0], axis=1)

            # Combine similarities with weights
            title_weight = 0.7
            short_desc_weight = 0.2
            desc_weight = 0.1

            data['combined_similarity'] += (title_weight * data['title_similarity'] +
                                            short_desc_weight * data['short_desc_similarity'] +
                                            desc_weight * data['desc_similarity']) * score

        # Sort by combined similarity
        data = data.sort_values(by='combined_similarity', ascending=False)

        # Sort again by date created
        data = data.sort_values(by='date_created', ascending=False)

        # Prioritize articles based on user preferences for categories
        if user_preferences:
            category_weights = user_preferences
            category_max_score = max(category_weights.values(), key=lambda x: x['score'])['score']
            for category, weight_info in category_weights.items():
                weight = weight_info['score']
                data.loc[data['category'] == category, 'combined_similarity'] *= (weight / category_max_score)

        # Sort again by adjusted combined similarity
        data = data.sort_values(by='combined_similarity', ascending=False)

        # Remove unnecessary columns
        unnecessary_columns = ['title_vector', 'short_desc_vector', 'desc_vector', 'title_similarity',
                                    'short_desc_similarity', 'desc_similarity']
        for column in unnecessary_columns:
            if column in data.columns:
                data = data.drop(columns=[column])


        # Extract the only 50 of id of the sorted data to be used for recommendation
        recommended_ids = data['id'][:50]

        return recommended_ids









############################################################################################################

    def preprocessing(self, text):
        def _remove_non_ascii(text):
            return ''.join(i for i in text if ord(i) < 128)
        
        def make_lower_case(text):
            return text.lower()
        
        def remove_stop_words(text):
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)
            return text
        
        def remove_html(text):
            html_pattern = re.compile('<.*?>')
            return html_pattern.sub(r'', text)
        
        def remove_punctuation(text):
            tokenizer = RegexpTokenizer(r'\w+')
            text = tokenizer.tokenize(text)
            text = " ".join(text)
            return text
        
        def remove_digits(text):
            return re.sub(r'\d+', '', text)
        
        def decode_unicode_escape(text):
            decoded_text = bytes(text, "utf-8").decode("unicode_escape")
            return decoded_text

        text = decode_unicode_escape(text)
        text = _remove_non_ascii(text)
        text = make_lower_case(text)
        text = remove_stop_words(text)
        text = remove_html(text)
        text = remove_punctuation(text)
        text = remove_digits(text)

        text = [x for x in text.split(' ') if x]

        #text = " ".join(text)

        return text
    
    

    def update_like_keywords(self, text, current_time, weight=0.1):
        text = self.preprocessing(text)

        print("text", text)
        
        for keyword in text:

            if keyword in self.user_keywords:
                self.user_keywords[keyword]['score'] += self.user_keywords[keyword]['score'] * weight
                self.user_keywords[keyword]['last_modified'] = current_time
                if self.user_keywords[keyword]['score'] > 5:
                    self.user_keywords[keyword]['score'] = 5
            else:
                self.user_keywords[keyword] = {'score': 2.5, 'last_modified': current_time}
        return self.user_keywords

    def update_like_categories(self, category, current_time, weight=0.1):
        if category in self.user_categories:
            self.user_categories[category]['score'] += self.user_categories[category]['score'] * weight
            self.user_categories[category]['last_modified'] = current_time
            if self.user_categories[category]['score'] > 5:
                self.user_categories[category]['score'] = 5
        else:
            self.user_categories[category] = {'score': 2.5, 'last_modified': current_time}
        return self.user_categories


    def update_dislike_keywords(self, text, current_time, weight=0.1):
        text = self.preprocessing(text)
        
        for keyword in text:
            if keyword in self.user_keywords:
                self.user_keywords[keyword]['score'] -= self.user_keywords[keyword]['score'] * weight
                self.user_keywords[keyword]['last_modified'] = current_time
                if self.user_keywords[keyword]['score'] < 0:
                    self.user_keywords[keyword]['score'] = 0
            else:
                self.user_keywords[keyword] = {'score': 2.5, 'last_modified': current_time}
        return self.user_keywords

    def update_dislike_categories(self, category, current_time, weight=0.1):
        if category in self.user_categories:
            self.user_categories[category]['score'] -= self.user_categories[category]['score'] * weight
            self.user_categories[category]['last_modified'] = current_time
            if self.user_categories[category]['score'] < 0:
                self.user_categories[category]['score'] = 0
        else:
            self.user_categories[category] = {'score': 2.5, 'last_modified': current_time}
        return self.user_categories


    def delete_overflow_keywords(self):
        if len(self.user_keywords) > 1000:
            sorted_keywords = sorted(self.user_keywords.items(), key=lambda x: x[1]['last_modified'], reverse=False)
            for keyword, data in sorted_keywords[:100]:
                del self.user_keywords[keyword]
        return self.user_keywords

    def delete_overflow_categories(self):
        if len(self.user_categories) > 1000:
            sorted_categories = sorted(self.user_categories.items(), key=lambda x: x[1]['last_modified'], reverse=False)
            for category, data in sorted_categories[:100]:
                del self.user_categories[category]
        return self.user_categories


    def update_view_history(self):
        article_history = self.user_view_history
        current_time = datetime.now().timestamp()

        #print(article_history)
        for article in article_history:
            category = article['category']

            # Update keywords
            self.update_like_keywords(article['title'], current_time, weight=0.1)

            # Update categories
            self.update_like_categories(category, current_time, weight=0.1)

        # Emppyt the view history
        article_history = []
                
        sorted_keywords = sorted(self.user_keywords.items(), key=lambda x: x[1]['last_modified'], reverse=True)
        sorted_categories = sorted(self.user_categories.items(), key=lambda x: x[1]['last_modified'], reverse=True)

        updated_keywords = OrderedDict()
        updated_categories = OrderedDict()

        for keyword, score in sorted_keywords:
            updated_keywords[keyword] = score

        for category, score in sorted_categories:
            updated_categories[category] = score

        self.user_keywords = updated_keywords
        self.user_categories = updated_categories

        updated_keywords = self.delete_overflow_keywords()
        updated_categories = self.delete_overflow_categories()

        self.user_keywords = updated_keywords
        self.user_categories = updated_categories
        
        #with open('data/user.json', 'w') as f:
        #    json.dump(self.user, f, indent=4)


    def set_feedback(self, article, feedback):
        current_time = datetime.now().timestamp()

        # Update keywords
        if feedback.lower() == 'like':
            if article not in self.user_likes:
                self.user_keywords = self.update_like_keywords(article['title'], current_time, weight=0.8)
        elif feedback.lower() == 'dislike':
            if article not in self.user_dislikes:
                self.user_keywords = self.update_dislike_keywords(article['title'], current_time, weight=0.8)

        # Update category
        if feedback.lower() == 'like':
            if article not in self.user_likes:
                self.user_categories = self.update_like_categories(article['category'], current_time, weight=0.8)
        elif feedback.lower() == 'dislike':
            if article not in self.user_dislikes:
                self.user_categories = self.update_dislike_categories(article['category'], current_time, weight=0.8)

        # Check if liked_articles/disliked_articles keys exist
        if 'liked_articles' not in self.user:
            self.user_likes = []
        if 'disliked_articles' not in self.user:
            self.user_dislikes = []

        # Memorize the liked/disliked article
        if feedback.lower() == 'like' and article not in self.user_likes:
            self.user_likes.append(article)
        elif feedback.lower() == 'dislike' and article not in self.user_dislikes:
            self.user_dislikes.append(article)
        else:
            print("Article already present in either like or dislike")

        # Sort user_keywords and user_categories by last modified time
        self.user_keywords = OrderedDict(sorted(self.user_keywords.items(), key=lambda x: x[1]['last_modified'], reverse=True))
        self.user_categories = OrderedDict(sorted(self.user_categories.items(), key=lambda x: x[1]['last_modified'], reverse=True))

        #with open('data/user_demo.json', 'w') as f:
        #    json.dump(self.user, f, indent=4)
        

