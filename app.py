from flask import Flask, render_template, request
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
data_path = 'updated_linkedin.csv'
df = pd.read_csv(data_path)

# Convert the clean_skills to a format usable by the model
df['clean_skills_str'] = df['clean_skills'].apply(
    lambda x: ' '.join(ast.literal_eval(x)) if pd.notna(x) else ''
)

# TF-IDF Vectorization of the skills
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['clean_skills_str'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(user_name, top_n=5):
    """
    Given a user name, this function returns top N recommended users based on skill similarity,
    along with their LinkedIn profile links.
    """
    if user_name not in df['Name'].values:
        return f"User '{user_name}' not found. Please enter a valid user name."
    
    # Find the index of the user in the dataframe
    user_index = df[df['Name'] == user_name].index[0]

    # Get similar users based on cosine similarity
    similar_users = list(enumerate(cosine_sim[user_index]))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:top_n+1]  # Top N similar users

    # Fetch the names and LinkedIn profile links of recommended users
    recommended_users = [
        (df.iloc[i[0]]['Name'], df.iloc[i[0]]['linkedin']) for i in similar_users
    ]

    return recommended_users

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_name = request.form.get('user_name')
    recommendations = get_recommendations(user_name, top_n=5)

    if isinstance(recommendations, str):
        # If the recommendation function returns an error message
        return render_template('index.html', error_message=recommendations)
    else:
        return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
