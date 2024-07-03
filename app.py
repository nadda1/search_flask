from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np 
import os 
# Initialize Flask app
app = Flask(__name__)
CORS(app)  


model = SentenceTransformer('SeyedAli/Multilingual-Text-Semantic-Search-Siamese-BERT-V1')

csv_file_path = os.path.join(os.path.dirname(__file__), 'job_data_v2.csv')
jobs = pd.read_csv(csv_file_path)


@app.route("/")
def home_view():
        return "<h1>Welcome to our app</h1>"
# Define the number of items per page
ITEMS_PER_PAGE = 10

@app.route('/search', methods=['POST'])
def search():
    # Get query and pagination parameters
    query = request.json.get('query')
    page = request.args.get('page', default=1, type=int)

    # Encode query and titles
    query_emb = model.encode(query)
    title_embs = model.encode(jobs['title'].tolist())

    # Calculate scores
    scores = util.dot_score(query_emb, title_embs)[0].cpu().tolist()

    # Filter based on threshold
    threshold = 0.6
    filtered_indices = [index for index, score in enumerate(scores) if score > threshold]

    # Apply pagination
    start_index = (page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    filtered_df = jobs.iloc[filtered_indices][start_index:end_index]
    filtered_df.replace({np.nan: None}, inplace=True)

    # Convert filtered DataFrame to dictionary
    filtered_dict = filtered_df.to_dict(orient='records')

    return jsonify({'search_jobs': filtered_dict})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
