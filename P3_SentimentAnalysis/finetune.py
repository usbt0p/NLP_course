from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

# Carga el dataset
dataset = load_dataset("imdb")

# TODO 1: Carga el tokenizador y el modelo a refinar
model_name = "prajjwal1/bert-tiny" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# TODO 2: Tokeniza el dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# TODO 3: Usa la clase TrainingArguments para definir las opciones de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# TODO 4: Implementa una función para obtener la precisión durante la evaluación, a partir del par (logits, labels)
# logits es una lista de tuplas con las puntuaciones para cada clase
# labels es una lista con las clases para cada ejemplo de evaluación
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# TODO 5: Usa la clase Trainer para definir un "entrenador" del modelo
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# TODO 6: Entrena/refina el modelo, evalúa su rendimiento y muestra su precisión
print("Iniciando entrenamiento...")
trainer.train()

print("\nEvaluando el modelo...")
eval_result = trainer.evaluate()
print(f"Precisión: {eval_result['eval_accuracy']:.4f}")
print(f"Loss: {eval_result['eval_loss']:.4f}")

# Guardar el modelo
trainer.save_model("./P3_SentimentAnalysis/fine_tuned_model")
print("Modelo guardado en P3_SentimentAnalysis/fine_tuned_model")
