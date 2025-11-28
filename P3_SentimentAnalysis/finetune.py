from datasets import load_dataset

# Carga el dataset
dataset = load_dataset("imdb")

# TODO 1: Carga el tokenizador y el modelo a refinar
model_name = "prajjwal1/bert-tiny"  # or "distilbert-base-uncased"
# tokenizer = ...
# model = ...

# TODO 2: Tokeniza el dataset

# TODO 3: Usa la clase TrainingArguments para definir las opciones de entrenamiento

# TODO 4: Implementa una función para obtener la precisión durante la evaluación, a partir del par (logits, labels)
# logits es una lista de tuplas con las puntuaciones para cada clase
# labels es una lista con las clases para cada ejemplo de evaluación

# TODO 5: Usa la clase Trainer para definir un "entrenador" del modelo

# TODO 6: Entrena/refina el modelo, evalúa su rendimiento y muestra su precisión
