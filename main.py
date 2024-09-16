import firebase_admin
from firebase_admin import credentials
from flask import Flask, request, jsonify
from firebase_admin import firestore
import recommender as Recommender
import numpy as np
import pandas as pd
from flask_cors import CORS, cross_origin

from flask_caching import Cache

cache = Cache()


# Initialize Firebase


cred = credentials.Certificate("private/ase-project-cd5cc-firebase-adminsdk-vhsw6-fb6268df8c.json")
firebase_admin.initialize_app(cred)

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Initialize the cache with your Flask app
cache.init_app(app)


@app.route('/get_documents', methods=['GET'])
def get_documents():
    # get documents from json file
    data = pd.read_json("data/bloomberg_quint_news.json")
    data['id'] = range(0, len(data))
    data = data.to_dict(orient='records')
    return jsonify(data)
    
@app.route('/get_users', methods=['GET'])
def get_users():
    with app.app_context():
        db = firestore.client()
        documents = db.collection('users').get()
        data = [doc.to_dict() for doc in documents]
        #print(data,"data")
        return jsonify(data)

@app.route('/get_personalized_news', methods=['GET'])
@cross_origin()
def personalized_news():
    print("Loading data")
    recommender.load_data(get_documents().json)
    print("Loaded data")
    print("Personalized news")
    user_id = request.args.get('userId')

    if not user_id:
        print("User id not provided")
        return jsonify({"error": "user_id is required"}), 400

    # Check if the recommendations are cached for this user ID
    cached_recommendations = cache.get(user_id)
    if cached_recommendations:
        print("returning cached data")
        return jsonify(cached_recommendations)

    with app.app_context():
        print("Connecting to firestore")
        db = firestore.client()
        print("Connected to firestore")

        # user_id = "DGINNGfXhIfQkJWnDoB51JbGtgU2"
        # Get user data
        user = db.collection('users').document(user_id).get()
        print("User data", user)
        if not user.exists:
            print("User not found")
            return jsonify({"error": "User not found"}), 404

        print("User found")
        # get user data
        user_data = user.to_dict()


        # print check if user_keywords is empty which is a type dict
        print("Checking if user_keywords is empty")
        if user_data['user_keywords'] == {}:
            print("User keywords is empty")
        # print if user_categories is empty
        print("Checking if user_categories is empty")
        if user_data['user_categories'] == {}:
            print("User categories is empty")

        # print if liked_articles is empty
        print("Checking if liked_articles is empty")
        if user_data['liked_articles'] == []:
            print("Liked articles is empty")

        # print if disliked_articles is empty
        print("Checking if disliked_articles is empty")
        if user_data['disliked_articles'] == []:
            print("Disliked articles is empty")
        # print if user_history is empty
        print("Checking if user_history is empty")
        if user_data['user_history'] == {}:
            print("User history is empty")


        # call recommender_news_based_on_keywords
        print("Getting personalised news")

        print("User", user_id)
        print("Keywords", user_data['user_keywords'])
        print("Categories", user_data['user_categories'])
        #print("Liked articles", user_data['liked_articles'])
        #print("Disliked articles", user_data['disliked_articles'])        
        # load data to recommender and user
        recommender.load_user(user_data)


        # Call the method for recommending news
        print("Returning recommended news")
        recommended_news_id = recommender.recommend_news_based_on_keywords_and_preferences() # get the id of the articles

        # Get the articles from the id
        recommended_news = [article for article in get_documents().json if article['id'] in recommended_news_id]

        # Cache the recommendations for this user ID
        cache.set(user_id, recommended_news)

        # return the recommended news
        return jsonify(recommended_news)

def convert_to_serializable(obj):
    # if DataFrame
    if isinstance(obj, pd.DataFrame):
        obj.columns = obj.columns.str.lower()
        return obj.to_dict(orient='records')

    try:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    except Exception as e:
        print("Error converting to serializable format:", e)
        return None


@app.route('/get_personalized_news_based_on_author', methods=['GET'])
def personalized_news_based_on_author():
    print("Loading data")
    recommender.load_data(get_documents().json)
    print("Loaded data")
    print("Personalized news")

    query = request.args.to_dict()
    user_id = query.get('userId')
    author = query.get('author')

    # Check if both user ID and author are provided
    if not user_id:
        return jsonify({"error": "userId is required"}), 400
    if not author:
        return jsonify({"error": "author is required"}), 400

    with app.app_context():
        print("Connecting to firestore")
        db = firestore.client()
        print("Connected to firestore")

        # Get user data
        user = db.collection('users').document(user_id).get()
        print("User data")
        if not user.exists:
            return jsonify({"error": "User not found"}), 404

        print("User found")
        # Get user data
        user_data = user.to_dict()

        # Get personalized news based on the author
        print("Getting personalised news")

        recommender.load_user(user_data)
        recommended_news = recommender.recommend_news_based_on_author(author=author)
        # Convert recommended_news to a serializable format
        recommended_news = convert_to_serializable(recommended_news)

        # Return the recommended news
        return jsonify(recommended_news)




@app.route('/get_recommended_news_based_on_keyword', methods=['GET'])
def recommend_news_based_on_keyword():
    print("Loading data")
    recommender.load_data(get_documents().json)
    print("Loaded data")

    print("Recommend news based on keyword")
    user_id = request.args.get('userId')
    keyword = request.args.get('keyword')

    if not user_id:
        print("User id not provided")
        return jsonify({"error": "user_id is required"}), 400
    if not keyword:
        print("Keyword not provided")
        return jsonify({"error": "keyword is required"}), 400
    
    with app.app_context():
        print("Connecting to firestore")
        db = firestore.client()
        print("Connected to firestore")

        # Get user data
        user = db.collection('users').document(user_id).get()
        print("User data")
        if not user.exists:
            print("User not found")
            return jsonify({"error": "User not found"}), 404

        print("User found")
        # get user data
        user_data = user.to_dict()

        # call recommender_news_based_on_keywords
        print("Getting personalised news")

        recommender.load_user(user_data)
        recommended_news = recommender.recommend_news_based_on_keyword(keyword)
        recommended_news = convert_to_serializable(recommended_news)

        # return the recommended news
        return jsonify(recommended_news)


# Add data to database
@app.route('/add_data', methods=['POST'])
def add_data():
    with app.app_context():
        db = firestore.client()
        data = request.json
        db.collection('documents').add(data)
        return jsonify({"message": "success"})


# Add new user to database
@app.route('/add_user', methods=['POST'])
def add_user():
    with app.app_context():
        db = firestore.client()
        data = request.json
        db.collection('users').add(data)
        return jsonify({"message": "success"})
    
def update_user_data_to_db(user_id, user_data):
    with app.app_context():
        db = firestore.client()
        db.collection('users').document(user_id).set(user_data)
        return jsonify({"message": "success"})


# Update user view history
@cross_origin()
@app.route('/update_user_view_history', methods=['GET'])
def update_user_view_history():
    with app.app_context():
        db = firestore.client()
        user_id = request.args.get('userId')

        print("user id", user_id)


        if not user_id:
            return jsonify({"error": "userId is required"}), 400

        # Get user data
        user = db.collection('users').document(user_id).get()
        
        if not user.exists:
            print("User not found")
            return jsonify({"error": "User not found"}), 404

        user_data = user.to_dict()
        print("User data", user_data)

        recommender.load_user(user_data)

        recommender.update_view_history()

        update_user_data_to_db(user_id, user_data)

        return jsonify({"message": "success"})


# set feedback for an article
@app.route('/set_feedback', methods=['POST'])
def set_feedback():

    with app.app_context():
        db = firestore.client()
        data = request.json
        print("Data:", data)

        # Get data from request")
        user_id = data.get('userId')
        article_title = data.get('articleTitle')
        feedback = data.get('feedback')

        print("User ID:", user_id)
        print("Article Title:", article_title)
        print("Feedback:", feedback)

        if not user_id:
            return jsonify({"error": "userId is required"}), 400
        if not article_title:
            return jsonify({"error": "articleTitle is required"}), 400
        if not feedback:
            return jsonify({"error": "feedback is required"}), 400

        # Get user data
        user = db.collection('users').document(user_id).get()
        print("User data")
        if not user.exists:
            print("User not found")
            return jsonify({"error": "User not found"}), 404

        user_data = user.to_dict()

        recommender.load_user(user_data)

        # retrieve the article metadata from the article title in the json file
        articles = get_documents().json
        article = [article for article in articles if article['title'] == article_title]

        if not article:
            return jsonify({"error": "Article not found"}), 404

        recommender.set_feedback(article[0], feedback)

        update_user_data_to_db(user_id, user_data)

        return jsonify({"message": "success"})






if __name__ == '__main__':
    print("Creating recommender object")
    recommender = Recommender.Recommender()

    print("Loading glove model")
    recommender.load_glove_model("gloves/glove.6B.50d.word2vec.txt")
    print("Loaded glove model")


    print("Starting flask server")

    app.run(debug=True, host='localhost', port=5001)

