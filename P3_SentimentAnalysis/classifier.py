import os
from sys import path
path.append(os.getcwd())

import numpy as np
np.random.seed(3141592)

import math
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from P2_Skipgram import p2_skipgram
from P1_Bpe.bpe import ByteLevelBPE

# 1: Implementa funciones para cargar token embeddings de un modelo entrenado y para obtener una 
# sola embedding por texto de entrada. Puedes usar la función de agregación que quieras. 

def load_embedding(path: str):
    # load embedding model and apply it to the input tex
    with open(path, 'r', encoding="utf-8") as e:
        header = e.readline()
        body = e.readlines()
        
    indexmode = body[0].split()[0].isdigit()

    if indexmode:
        data = {}
        for line in body:
            parts = line.strip().split()
            idx = int(parts[0])
            vec = [float(x) for x in parts[1:]]
            data[idx] = vec
        body = data
    else:
        # tab separation if the start is not a digit
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

# use the mean to compute a single embedding
# this will lose the order of the words but it's easier to do
# TODO other aggregations that are better /( loose less information)
def aggregate_single_embedding(embedding):
    # TODO maybe remove stopwords here, since we're already removing info,
    # in favour of simplicity, better to remove from the mean the non important stuff
    if not embedding: return None
    # axis=0 para mantener la dimensionalidad del vector
    return np.mean(embedding, axis=0)

class LogisticRegression:

    # * Campos para almacenar los pesos, sesgo y learning rate. 
    def __init__(self, num_weights):
        # TODO comprobar que los pesos puedan ser cualquier random o si tienen que ser ceros
        self.W = np.random.uniform(-0.2, 0.2, num_weights).astype(np.float32)
        self.b = 0.0
        self.lr = 0.01 # TODO ajustar
        self.losses = [] 
    
    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    # * Método `forward`, que implemente la combinación lineal de los pesos y sesgo con la entrada seguida de la función logística.
    def forward(self, input):
        z = np.dot(self.W.T, input) + self.b
        return self.sigmoid(z)

    # * Método `backward` que, dada la entrada, la salida obtenida y la salida deseada, modifique los parámetros del modelo.
    def backward(self, predicted, real, inputs):        
        # https://cs230.stanford.edu/fall2018/section_files/section3_soln.pdf
        # by the chain rule of ΔL/Δw = ΔL/Δsigmoid * Δsigmoid/Δ(w*x+b) * Δ(w*x+b)/Δw = (ŷ - y) @ X
        # and for biases: ΔL/Δb = ... = ŷ - y
        error = predicted - real
        m = real.shape[1]

        # we do that and account for the number of elems
        new_weights = np.dot(inputs, error.T) * (1 / m)
        new_bias = np.sum(error) * (1 / m)

        return new_weights.flatten(), new_bias
        
    # * Método `compute_loss`, que implemente la función de entropía cruzada binaria.
    def compute_loss(self, predicted, real) -> float:
        m = real.shape[1]
        epsilon = 1e-9
        loss = - (1/m) * np.sum(real * np.log(predicted + epsilon) + (1 - real) * np.log(1 - predicted + epsilon))
        return loss
        
    # * Método `fit`, que recibe los datos de entrenamiento y optimiza el modelo mediante descenso de gradiente.
    def fit(self, X, Y, epochs=100, interactive=False):
        
        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            # fordward pass
            activations = self.forward(X)
            
            # calcular perdida
            loss = self.compute_loss(activations, Y)
            self.losses.append(loss)
            pbar.set_description(f"Epoch {epoch} | Loss: {loss:.4f}")
            
            # backward pass, devolver nuevos W y b
            new_w, new_b = self.backward(activations, Y, X)
            
            # actualizar pesos con descenso de gradiente
            self.W = self.W - self.lr * new_w
            self.b = self.b - self.lr * new_b

        if interactive:
            plt.plot(self.losses)
            plt.title("Training Loss")
            plt.show()
            
    # Método `predict`, que usa `forward` y obtiene la salida final de inferencia en `{0, 1}`.
    def predict(self, input):
        output = self.forward(input)
        y_hat = output > 0.5
        return y_hat, output

def load_dataset(path):
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

def process_data(file_path):
    '''Auxiliary function to process a TSV into matrices of X and Y'''
    texts, labels = load_dataset(file_path)
    X_list, Y_list = [], []
    
    print(f"Procesando {file_path}...")
    for text, label in zip(texts, labels):
        # extract vecs from embeddings
        token_data = embeddings_from_tokens(text, embeddings, bpe_file)
        vectors = [v for k, v in token_data]
        
        avg_vec = aggregate_single_embedding(vectors)
        assert avg_vec is not None
        
        X_list.append(avg_vec)
        Y_list.append(label)
    
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
    X_train, Y_train = process_data(train_file)
    # TODO data augmentation?

    # 2. actually train
    print(f"Entrenando con {X_train.shape[1]} muestras...")
    num_features = X_train.shape[0]
    model = LogisticRegression(num_weights=num_features)
    model.fit(X_train, Y_train, epochs=1000, interactive=True)
    
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
    