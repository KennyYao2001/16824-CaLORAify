from sentence_transformers import SentenceTransformer
from parse_output import parse_ingredients
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# first, we need to get all the ingredients from database and their embeddings
with open("database.json", "r") as f:
    database = json.load(f)

ingredients = list(database.keys()) # 1597

calories = [database[ingredient]["calorie"] for ingredient in ingredients] # 1597
embeddings = model.encode(ingredients) # 1597 * 384
embeddings = np.array(embeddings)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # 1597 * 384

# second, we need to get the embeddings of the ingredients in the output
with open("test_result.json", "r") as f:
    output = json.load(f)

for idx in range(100,110):
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
        # print(ingredients[most_similar_idx])
        # print(calories[most_similar_idx])
        total_calories += calories[most_similar_idx] * amount
        print(amount, ingredient_name, ingredients[most_similar_idx], calories[most_similar_idx])
    print(output[idx]["pred"])
    print("Final Calories: ", total_calories)
    print()
