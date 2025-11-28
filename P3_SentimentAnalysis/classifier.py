import numpy as np
import pickle

# TODO 1: Implementa funciones para cargar token embeddings de un modelo entrenado y para obtener una sola embedding por texto de entrada. Puedes usar la función de agregación que quieras. 

# TODO 2: Implementa la clase LogisticRegression con los siguientes componentes:
# * Campos para almacenar los pesos, sesgo y learning rate.
# * Método `forward`, que implemente la combinación lineal de los pesos y sesgo con la entrada seguida de la función logística.
# * Método `backward` que, dada la entrada, la salida obtenida y la salida deseada, modifique los parámetros del modelo.
# * Método `compute_loss`, que implemente la función de entropía cruzada binaria.
# * Método `fit`, que recibe los datos de entrenamiento y optimiza el modelo mediante descenso de gradiente.
# # Método `predict`, que usa `forward` y obtiene la salida final de inferencia en `{0, 1}`.

# TODO 3: Implementa una función principal que realice todos los pasos necesarios para entrenar y evaluar un modelo de regresión logística que usa agregados de token embeddings como características.

# NOTA: no es necesario almacenar el modelo de regresión.