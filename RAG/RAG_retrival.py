from sentence_transformers import SentenceTransformer
from parse_output import parse_ingredients
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

def save_result_graph(predicted_calories, image_path, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Create a new figure with a specific size
    plt.figure(figsize=(10, 8))
    
    # Load and display the image
    img = plt.imread(image_path)
    plt.imshow(img)
    
    # Set the title with predicted calories
    plt.title(f'Predicted Calories: {predicted_calories:.1f}', pad=20)
    
    # Remove axes for cleaner visualization
    plt.axis('off')
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# first, we need to get all the ingredients from database and their embeddings
with open("database.json", "r") as f:
    database = json.load(f)

ingredients = list(database.keys()) # 1597

calories = [database[ingredient]["nrg"] for ingredient in ingredients] # 1597
embeddings = model.encode(ingredients) # 1597 * 384
embeddings = np.array(embeddings)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # 1597 * 384

# second, we need to get the embeddings of the ingredients in the output
with open("test_result.json", "r") as f:
    output = json.load(f)


predicted_calories = []
for idx in range(100):
    current_ingredients = parse_ingredients(output[idx]["pred"])
    total_calories = 0
    for amount, ingredient_name in current_ingredients:
        # Sentences are encoded by calling model.encode()
        embedding = model.encode(ingredient_name)
        embedding = np.array(embedding)
        embedding = embedding / np.linalg.norm(embedding) # 384,
        # calculate the cosine similarity between the embedding and all the embeddings in the database
        similarities = np.dot(embeddings, embedding) # 1597,
        # get the index of the most similar embedding
        most_similar_idx = np.argmax(similarities)
        total_calories += calories[most_similar_idx] * amount

        print(amount, ingredients[most_similar_idx], calories[most_similar_idx])
    predicted_calories.append(total_calories)

    
    img_path = output[idx]['image_path']
    # print(img_path)
    print(idx)
    print("Predicted Calories: ", total_calories)
    save_path = "/root/autodl-tmp/VLR_project/RAG/saved_graphs/" + img_path.split("/")[-1]
    save_result_graph(total_calories, img_path, save_path)
    