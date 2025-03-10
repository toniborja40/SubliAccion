# Este archivo, se utiliza para entrenar el modelo de inteligencia artificial que será utilizado en el chatbot.
# Desarrollado por José Pulido
import json
import pickle
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Inicializar lematizador
lemmatizer = WordNetLemmatizer()

# Cargar datos de intents.json
with open("model/intents.json") as file:
    intents = json.load(file)

words = []  # Lista de palabras únicas
classes = []  # Lista de etiquetas únicas
documents = []
ignore_words = ["?", "!", ".", ","]  # Palabras a ignorar

# Procesar cada intención en intents.json
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)  # Tokenización
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lematizar palabras y eliminar duplicados
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Guardar words.pkl y classes.pkl
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Crear datos de entrenamiento
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
    
    for w in words:
        bag.append(1 if w in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Mezclar datos y convertir a numpy
random.shuffle(training)
training = np.array(training, dtype=object)

# Separar X (input) e Y (output)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Construcción del modelo
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compilar modelo
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

# Entrenar modelo
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Guardar modelo entrenado
model.save("chatbot_model.h5")

print("✅ ¡Entrenamiento completado! Modelo guardado como chatbot_model.h5")
