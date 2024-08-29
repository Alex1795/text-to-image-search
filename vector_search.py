import torch
from transformers import CLIPProcessor, CLIPModel

# Initialize CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
model.eval()


def create_text_embedding(text):
    """
    Create an embedding for the given text using the CLIP model.

    Args:
    text (str): The input text to embed.

    Returns:
    numpy.ndarray: The embedding vector for the input text.
    """
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.numpy().flatten()



# To test the embedding creation
#print(str(create_text_embedding("people playing in the park")).replace("  "," ").replace(" ",','))