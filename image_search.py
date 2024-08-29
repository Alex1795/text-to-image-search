import streamlit as st
from elasticsearch import Elasticsearch
import pandas as pd
from vector_search import create_text_embedding
from PIL import Image
import os

# Load sensitive information from environment variables
ELASTIC_CLOUD_ID = os.environ.get("ELASTIC_CLOUD_ID")
ELASTIC_API_KEY = os.environ.get("ELASTIC_API_KEY")
INDEX_NAME = os.environ.get("ELASTIC_INDEX_NAME", "flickr30k_embeddings-clip")
# Streamlit app title
st.title("Image search powered by vector similarity")



# Connect to Elasticsearch
es = Elasticsearch(
    cloud_id=ELASTIC_CLOUD_ID,
    api_key=ELASTIC_API_KEY
)

# Search query input
search_query = st.text_input("Enter your search query:")

if search_query:
    # Perform the search
    result = es.search(
        index=INDEX_NAME,
        body={
  "_source": ["image_id"],
  "knn": {
    "field": "embedding",
    "k": 6,
    "num_candidates": 5000,
    "query_vector": list(create_text_embedding(search_query))
  }
}
    )

    # Extract and format the results
    hits = result['hits']['hits']
    # Extract and format the results
    hits = result['hits']['hits']
    if hits:
        # Display images in a grid
        st.subheader("Search Results:")
        cols = st.columns(3)  # Adjust the number of columns as needed
        for i, hit in enumerate(hits):
            image_id = hit['_source']['image_id']
            image_url = f"/Images/{image_id}.jpg"
            try:
                # If images are served from a web server
                #response = requests.get(image_url)
                image = Image.open(f"./Images/{image_id}.jpg")

                # If images are served locally, use this instead:
                # image = Image.open(f"./images/{image_id}.jpg")

                cols[i % 3].image(image, caption=f"Image {image_id}", use_column_width=True)
            except Exception as e:
                cols[i % 3].error(f"Error loading image {image_id}: {str(e)}")

        # Create a DataFrame
        df = pd.DataFrame([hit['_source'] for hit in hits])

        # Display results as a table
        st.subheader("Image IDs:")
        st.table(df)
    else:
        st.info("No results found.")

# Add a note about connection status
if es.ping():
    st.success("Connected to Elasticsearch")
else:
    st.error("Failed to connect to Elasticsearch")