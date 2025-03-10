# Este archivo, se utiliza para manejar la lógica de la aplicación y proporcionar una interfaz interactiva para el chatbot
# Desarrollado por José Pulido
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import json
import nltk
from nltk.stem import WordNetLemmatizer
import random
from unidecode import unidecode  # Para manejar caracteres especiales

# Inicializar lematizador
lemmatizer = WordNetLemmatizer()

# Cargar archivos con manejo de errores
try:
    model = load_model("chatbot_model.h5")  # Asegúrate de que el modelo existe
    words = pickle.load(open("words.pkl", "rb"))  # Lista de palabras
    classes = pickle.load(open("classes.pkl", "rb"))  # Lista de clases
    with open("model/intents.json", "r", encoding="utf-8") as file:
        intents = json.load(file)  # Intenciones cargadas correctamente
except FileNotFoundError as e:
    st.error(f"⚠️ Archivo no encontrado: {e.filename}. Verifica que el modelo y los datos existen.")
    st.stop()

# Función para limpiar y procesar una oración
def clean_up_sentence(sentence):
    sentence = sentence.lower()
    sentence = unidecode(sentence)  # Convierte caracteres especiales sin perder significado
    sentence_words = nltk.word_tokenize(sentence)  # Tokeniza la oración
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]  # Lematiza palabras
    return sentence_words

# Función para convertir la entrada en un "Bag of Words"
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  # Vector de ceros con el tamaño correcto
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array([bag])  # Retorna un array con forma (1, len(words))

# Función para predecir la intención del usuario
def predict_intent(text):
    p = bow(text, words)
    res = model.predict(p)[0]  # Predice la intención
    ERROR_THRESHOLD = 0.25  # Umbral de confianza mínima
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Filtra resultados confiables
    results.sort(key=lambda x: x[1], reverse=True)  # Ordena por mayor probabilidad

    if results:
        return classes[results[0][0]]  # Devuelve la intención más probable
    return "unknown"  # Devuelve "desconocido" si no encuentra una intención

# Función para obtener la respuesta del chatbot
def chatbot_response(text):
    intent = predict_intent(text)
    for i in intents["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])  # Devuelve una respuesta aleatoria de la intención
    return "Lo siento, no entendí tu pregunta."

# 🎨 **Interfaz mejorada con Streamlit**
st.title("🤖 Asistente Virtual de SubliAcción")
st.markdown("💬 **Bienvenido a SubliAcción** - Personalizamos con creatividad y calidad. ¿En qué podemos ayudarte hoy?")
st.markdown("Chatbot desarrollado por José Antonio Pulido Colmenares")

# Manejo del historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
prompt = st.chat_input("Escribe tu mensaje aquí...")
if prompt:
    # Mostrar mensaje del usuario en la interfaz
    with st.chat_message("user"):
        st.markdown(f"**Tú:** {prompt}")
    st.session_state.messages.append({"role": "user", "content": f"**Tú:** {prompt}"})

    # Obtener la respuesta del chatbot
    response = chatbot_response(prompt)

    # Mostrar la respuesta en la interfaz
    with st.chat_message("assistant"):
        st.markdown(f"🤖 **SubliAcción:** {response}")
    st.session_state.messages.append({"role": "assistant", "content": f"🤖 **SubliAcción:** {response}"})
