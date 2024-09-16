# News Article Recommendation Server

This repository contains the backend server implementation for a personalized news article recommendation system using Flask, Firebase, and a machine learning model. The system provides endpoints to retrieve, add, and modify user and document data, and it generates personalized news recommendations based on user preferences and behaviors.

## Features

- Fetch and display news documents.
- User management (add, fetch, and update user data).
- Personalized news recommendations based on user behavior and preferences.
- Feedback mechanisms for user interactions with recommended articles.

## Prerequisites

- Python 3.8 or higher
- Firebase project and admin credentials
- Flask and additional Python libraries as listed in `requirements.txt`
- Firebase Configuration:
    Obtain your Firebase service account key file from the Firebase console.
    Place the key file in the private/ directory.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repository/news-recommendation-server.git
   cd news-recommendation-server
## API Endpoints
1. GET /get_documents: Fetches all documents.
2. GET /get_users: Retrieves all users.
3. GET /get_personalized_news: Provides personalized news based on the user's history and preferences.
4. POST /add_user: Adds a new user to the system.
5. POST /set_feedback: Submits user feedback for an article.
