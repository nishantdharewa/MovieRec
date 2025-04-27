from flask import Flask, render_template, request, redirect, url_for,session
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity

app = Flask(__name__)


# Load the model data from the pickle file
with open('recommendation_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

movies_df = model_data['movies']
ratings = model_data['ratings']
user_movie_matrix = model_data['user_movie_matrix']
user_similarity_df = model_data['user_similarity_df']


def recommend_movies(user_id, num_recommendations=5):
    if user_id not in user_movie_matrix.index:
        return None  

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    user_ratings = user_movie_matrix.mul(similar_users, axis=0).sum(axis=0)
    user_ratings /= similar_users.sum()

    recommendations = user_ratings[user_movie_matrix.loc[user_id] == 0]
    recommendations = recommendations.sort_values(ascending=False).head(num_recommendations)

    return recommendations


app.secret_key = 'supersecretkey'

# User authentication for simplicity
USER_CREDENTIALS = {"Username": "nishant", "Password": "123"}

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        Username = request.form.get("Username")  # Match with input field name in login.html
        Password = request.form.get("Password")  # Match with input field name in login.html
        if Username == USER_CREDENTIALS["Username"] and Password == USER_CREDENTIALS["Password"]:
            return redirect(url_for("home"))
        else:
            return "Invalid Credentials, Please try again."
    return render_template("login.html")
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))




    
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validate the form data
        if password != confirm_password:
            return "Passwords do not match.", 400
        
        # Process the registration (add user to database, etc.)
        # For simplicity, we just print the data (you'd save to a database in real-world apps)
        print(f"User registered: {name}")

        # Redirect to the home page after registration
        return redirect(url_for('home'))
    
    return render_template('register.html') 


@app.route("/home")
def home():
    return render_template("home.html")


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
    elif request.method == 'GET':
        user_id = int(request.args.get('user_id'))

    recommendations = recommend_movies(user_id)

    if recommendations is None:
        return redirect(url_for('new_user', user_id=user_id))
    else:
        recommended_movies_series = recommendations
        recommended_movies_series.name = 'score'
        recommended_movie_ids = recommended_movies_series.index.tolist()
        recommended_movies = movies_df[movies_df['movieId'].isin(recommended_movie_ids)][['movieId', 'title']]
        recommended_movies = recommended_movies.merge(recommended_movies_series, left_on='movieId', right_index=True)
        return render_template("recommend.html", movies=recommended_movies[['title', 'score']].to_dict(orient='records'))

@app.route('/new_user/<int:user_id>')
def new_user(user_id):
    initial_movies = movies.sample(10)
    return render_template("new_user.html", user_id=user_id, movies=initial_movies.to_dict(orient='records'))


@app.route('/submit_ratings', methods=['POST'])
def submit_ratings():
    user_id = int(request.form['user_id'])
    initial_ratings = {int(movie_id): float(rating) for movie_id, rating in request.form.items() if movie_id != 'user_id'}

    new_user_ratings = pd.Series(0.0, index=user_movie_matrix.columns)
    for movie_id, rating in initial_ratings.items():
        new_user_ratings[movie_id] = float(rating)
    user_movie_matrix.loc[user_id] = new_user_ratings

    updated_similarity = cosine_similarity(user_movie_matrix)  # This line uses the imported cosine_similarity
    global user_similarity_df
    user_similarity_df = pd.DataFrame(updated_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    # Re-save the updated model data after the new user interaction
    model_data['user_movie_matrix'] = user_movie_matrix
    model_data['user_similarity_df'] = user_similarity_df
    with open('recommendation_model.pkl', 'wb') as file:
        pickle.dump(model_data, file)

    return redirect(url_for('recommend', user_id=user_id))


@app.route('/movies')
def movies():
    return render_template('movies.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
 
    app.run(debug=True)