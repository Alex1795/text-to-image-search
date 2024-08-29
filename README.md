# Text to image search

This project creates embeddings for images using the CLIP model and uploads them to Elasticsearch. Then it fires up a front end on streamlit to search images from a text query
It uses the Flickr30k dataset but can be adapted for other image datasets easily.


# Features
- Creates embeddings using the CLIP model
- Uploads them to Elasticsearch
- Creates a local front end to search and display the resulting images

# Usage

Run the upload script with:

```
python upload_images.py
```

Start the search front end using:

```
streamlit run image_search.py
```