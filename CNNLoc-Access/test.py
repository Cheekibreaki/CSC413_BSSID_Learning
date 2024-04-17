import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)

# Random embeddings for 4 words
embeddings = {
    "the": np.random.rand(3),
    "quick": np.random.rand(3),
    "brown": np.random.rand(3),
    "fox": np.random.rand(3)
}

# Transformation matrices for query, key, value
Q = np.random.rand(3, 3)
K = np.random.rand(3, 3)
V = np.random.rand(3, 3)

# Compute query, key, value for each word
queries = {word: np.dot(Q, emb) for word, emb in embeddings.items()}
keys = {word: np.dot(K, emb) for word, emb in embeddings.items()}
values = {word: np.dot(V, emb) for word, emb in embeddings.items()}

# Compute attention for each word
outputs = {}
for word in embeddings.keys():
    scores = np.array([np.dot(queries[word], keys[other]) for other in embeddings.keys()])
    attn_weights = softmax(scores)
    output = sum(attn_weights[i] * values[other] for i, other in enumerate(embeddings.keys()))
    outputs[word] = output

# Display the results
print("Attention Weights and Outputs for each word:")
for word, output in outputs.items():
    print(f"{word}: {output}")