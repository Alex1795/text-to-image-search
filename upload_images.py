import torch
from torchvision import transforms
from PIL import Image
import os
from transformers import CLIPModel, CLIPProcessor
from elasticsearch import Elasticsearch
from tqdm import tqdm


# Load sensitive information from environment variables
ELASTIC_CLOUD_ID = os.environ.get("ELASTIC_CLOUD_ID")
ELASTIC_API_KEY = os.environ.get("ELASTIC_API_KEY")
INDEX_NAME = os.environ.get("ELASTIC_INDEX_NAME", "flickr30k_embeddings-clip")


# Initialize CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
model.eval()

# Set up image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize Elasticsearch client
es = Elasticsearch(cloud_id=ELASTIC_CLOUD_ID, api_key=ELASTIC_API_KEY, timeout=500, max_retries=3,  # Add retries
    retry_on_timeout=True)


# Function to create embedding from an image
def create_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs.numpy().flatten()


# Function to upload to Elasticsearch
def upload_to_elasticsearch(image_id, embedding):
    doc = {
        'image_id': image_id,
        'embedding': embedding.tolist()
    }
    es.index(index=INDEX_NAME, id=image_id, body=doc)


# Main processing loop
flickr30k_dir = 'images/'  # Replace with your Flickr30k dataset path

for image_file in tqdm(os.listdir(flickr30k_dir)):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(flickr30k_dir, image_file)
        image_id = os.path.splitext(image_file)[0]

        embedding = create_embedding(image_path)
        upload_to_elasticsearch(image_id, embedding)

print("Processing complete. All embeddings uploaded to Elasticsearch.")