import os
from sys import path
path.append(os.getcwd())

import numpy as np
np.random.seed(3_141592) # π

import matplotlib.pyplot as plt
from tqdm import tqdm

from P1_Bpe.bpe import ByteLevelBPE

# --- HARDCODED STOPWORDS ---
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "did", "do", "does", "doing", "don", "down", "during",
    "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "s", "same", "she", "should", "so", "some", "such",
    "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up",
    "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "you", "your", "yours", "yourself", "yourselves"
}

def remove_stopwords(text: str) -> str:
    words = text.split()
    filtered_words = [w for w in words if w.lower() not in STOPWORDS]
    return " ".join(filtered_words)

# 1: Implementa funciones para cargar token embeddings de un modelo entrenado y para obtener una 
# sola embedding por texto de entrada. Puedes usar la función de agregación que quieras. 

def load_embedding(path: str):
    """Load embeddings from a file with specific format"""
    with open(path, 'r', encoding="utf-8") as e:
        header = e.readline()
        body = e.readlines()
        
    indexmode = body[0].split()[0].isdigit()

    # space separation if the start is a token id
    if indexmode:
        data = {}
        for line in body:
            parts = line.strip().split()
            idx = int(parts[0])
            vec = [float(x) for x in parts[1:]]
            data[idx] = vec
        body = data
    else:
        # tab separation if the start is a token (not it's id)
        body = {line[:line.find("\t")] : 
                [float(d) for d in line[line.find("\t")+1:].split(" ")]
                 for line in body}
            
    return header, body

def embeddings_from_tokens(input_text : str, 
                           embeddings : dict, 
                           bpe_model: str, 
                           mode="idx"):
    # make tokens
    bpe = ByteLevelBPE()
    bpe.logging = False
    bpe.load(bpe_model)
    tokens = bpe.encode(input_text) if mode == "idx" else bpe.tokenize(input_text)
    
    result = []
    for token in tokens:
        if token in embeddings:
            result.append((token, embeddings[token]))
    return result

def aggregate_single_embedding(embedding):
    """Use the mean to compute a single embedding
    this will loose the order of the words but it's easier to do
    """
    return np.mean(embedding, axis=0) # axis=0 does the mean over the columns

class LogisticRegression:
 
    def __init__(self, num_weights):
        self.W = np.random.uniform(-0.1, 0.1, num_weights).astype(np.float32)
        self.b = 0.0
        self.lr = 0.01
        self.losses = [] 
    
    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def forward(self, input):
        """Forward is the linear combination betwee weights and biases 
        and the input followed by activation."""
        z = np.matmul(self.W.T, input) + self.b
        return self.sigmoid(z)

    def backward(self, predicted, real, inputs):  
        """No autograd, so manual backpropagation:      
            - By the chain rule of `ΔL/Δw =`

                `= ΔL/Δsigmoid * Δsigmoid/Δ(w*x+b) * Δ(w*x+b)/Δw = (ŷ - y) @ X`
            - For biases it's easier: `ΔL/Δb = ... = ŷ - y`
        
            https://cs230.stanford.edu/fall2018/section_files/section3_soln.pdf
        """
        error = predicted - real
        m = real.shape[1]

        # (1 / m) to account for the number of training examples
        new_weights = np.matmul(inputs, error.T) * (1 / m)
        new_bias = np.sum(error) * (1 / m)

        return new_weights.flatten(), new_bias
        
    def compute_loss(self, predicted, real) -> float:
        """Compute binary cross entropy function. Preds should be activations"""
        m = real.shape[1]
        epsilon = 1e-9
        loss = - (1/m) * np.sum(real * np.log(predicted + epsilon) + (1 - real) * np.log(1 - predicted + epsilon))
        return loss
    
    def predict(self, input):
        """Get the final inference in `{0, 1}`."""
        output = self.forward(input)
        y_hat = output > 0.5
        return y_hat, output
        
    def fit(self, X, Y, epochs=100, interactive=False):
        """Update weights via gradient descent."""
        
        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            # fordward pass
            activations = self.forward(X)
            
            # loss
            loss = self.compute_loss(activations, Y)
            self.losses.append(loss)
            pbar.set_description(f"Epoch {epoch} | Loss: {loss:.4f}")
            
            # backward pass, return new W and b
            new_w, new_b = self.backward(activations, Y, X)
            
            # gradient descent
            self.W = self.W - self.lr * new_w
            self.b = self.b - self.lr * new_b
        
        if interactive:
            plt.plot(self.losses)
            plt.title("Training Loss")
            plt.show()
            plt.savefig("bce_loss.png")
            
def load_dataset(path):
    """Helper function to load a `.tsv` with data and labels."""
    texts, labels = [], []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    texts.append(parts[0])
                    labels.append(int(parts[1]))
    except FileNotFoundError:
        print(f"No se encontró el archivo {path}")
    return texts, labels

# 3: Implementa una función principal que realice todos los pasos necesarios para entrenar 
# y evaluar un modelo de regresión logística que usa agregados de token embeddings como características.
# NOTA: no es necesario almacenar el modelo de regresión.

def process_data(file_path, apply_stopwords=True):
    '''Auxiliary function to process a TSV into matrices of X and Y'''
    texts, labels = load_dataset(file_path)
    X_list, Y_list = [], []
    
    print(f"Procesando {file_path}...")
    acc = 0
    for text, label in zip(texts, labels):
        
        if apply_stopwords:
            temp = len(text.split())
            text = remove_stopwords(text)
            acc += temp - len(text.split())

        # extract vecs from embeddings
        token_data = embeddings_from_tokens(text, embeddings, bpe_file)
        vectors = [v for k, v in token_data]
        
        avg_vec = aggregate_single_embedding(vectors)
        assert avg_vec is not None
        
        X_list.append(avg_vec)
        Y_list.append(label)
    
    if apply_stopwords: print(f"Removed {acc} stopwords.")
    # transpose to have (Features, Samples)
    return np.array(X_list).T, np.array(Y_list).reshape(1, -1)

if __name__ == "__main__":

    basepath = "/home/usbt0p/Uni/PLN/PLN"
    embeddings_file = os.path.join(basepath, "P2_Skipgram/embeddings.txt")
    bpe_file = os.path.join(basepath, "P1_Bpe/bpe_model_1000.pkl")
    train_file = os.path.join(basepath, "P3_SentimentAnalysis/train.tsv")
    test_file = os.path.join(basepath, "P3_SentimentAnalysis/test.tsv")

    print("Cargando embeddings...")
    h, embeddings = load_embedding(embeddings_file)
    
    # 1. load train
    X_train, Y_train = process_data(train_file, apply_stopwords=True)
    # TODO data augmentation?

    # 2. actually train
    print(f"Entrenando con {X_train.shape[1]} muestras...")
    num_features = X_train.shape[0]
    model = LogisticRegression(num_weights=num_features)
    model.fit(X_train, Y_train, epochs=200, interactive=True)
    
    # 3. test
    print("-" * 30)
    print("Evaluando sobre Test set...")
    X_test, Y_test = process_data(test_file)
    predictions, probabilities = model.predict(X_test)
    
    # simple accuracy compute
    correct = np.sum(predictions == Y_test)
    total = Y_test.shape[1]
    accuracy = correct / total
    
    print(f"Accuracy en Test: {accuracy:.4f} ({correct}/{total})")
    