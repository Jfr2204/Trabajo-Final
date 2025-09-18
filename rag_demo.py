import re
import uuid
import os
from dotenv import load_dotenv
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import chromadb

# Cargar variables desde .env
load_dotenv()

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DB = os.getenv("CHROMA_DB")

# Conectar a Chroma Cloud
client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DB
)

collection = client.get_or_create_collection("document_chunks")

# Inicializar modelo y embeddings
llm = Llama(model_path="./models/mistral-7b-instruct-v0.2.Q6_K.gguf")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# Función para dividir texto en chunks
def dividir_en_chunks(texto, tamaño=300, solapamiento=50):
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + tamaño
        chunk = texto[inicio:fin]
        chunks.append(chunk)
        inicio += tamaño - solapamiento
    return chunks


# Preguntar al usuario si quiere cargar el dataset
cargar = input("¿Deseas cargar marfan.txt en Chroma? (s/n): ").strip().lower()

if cargar == "s":
    with open("data/marfan.txt", "r", encoding="utf-8") as f:
        texto_marfan = f.read()

    chunks = dividir_en_chunks(texto_marfan)
    embeddings = embed_model.encode(chunks).tolist()

    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"fuente": "marfan.txt"} for _ in chunks]
    )

    print(f"Se guardaron {len(chunks)} chunks en Chroma")
else:
    print("No se cargó marfan.txt")


# Función de búsqueda en Chroma
def buscar_chunks(query, top_k=2):
    q_embedding = embed_model.encode([query]).tolist()
    resultados = collection.query(
        query_embeddings=q_embedding,
        n_results=top_k
    )
    return resultados["documents"][0]


# Bucle de preguntas al modelo
print("\nEscribe tu pregunta (o 'salir' para terminar):")
while True:
    query = input("Pregunta: ").strip()
    if query.lower() == "salir":
        print("Saliendo del programa...")
        break

    relevant_chunks = buscar_chunks(query, top_k=2)

    prompt = "Usando solo la información a continuación, responde la pregunta.\n\n"
    for chunk in relevant_chunks:
        prompt += chunk + "\n"
    prompt += f"\nPregunta: {query}\nRespuesta:"

    respuesta = llm(prompt, max_tokens=200)
    print("\nRespuesta del modelo:")
    print(respuesta["choices"][0]["text"])
    print("-" * 60)



