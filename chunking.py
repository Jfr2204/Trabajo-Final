import re

# Abrimos el archivo de texto y leemos todo el contenido en la variable `doc`
with open('./data/marfan.txt') as file:
    doc = file.read()

# Reemplazamos los saltos de línea por espacios (para que no corten las oraciones)
doc_clean = doc.replace("\n", " ")

# Dividimos el texto en oraciones cada vez que aparece un punto, signo de pregunta,
# signo de exclamación o dos puntos, seguido de espacio(s).
single_sentence_list = re.split(r'(?<=[.:?!])\s+', doc_clean)
print (f"{len(single_sentence_list)} oraciones fueron encontradas")

# Creamos una lista de diccionarios, cada uno con:
# - la oración original ('sentence')
# - su índice dentro de la lista ('index')
sentences = [{'sentence': x, 'index': i} for i, x in enumerate(single_sentence_list)]


# Función que combina cada oración con las oraciones anteriores y posteriores
# según el tamaño del buffer (buffer_size).
def combine_sentences(sentences, buffer_size = 1):
    # Recorremos todas las oraciones por índice
    for i in range(len(sentences)):
        
        combined_sentences = ''  # inicializamos la cadena combinada vacía

        # Agregamos las oraciones anteriores (según el buffer)
        for j in range(i - buffer_size, i):
            if j >= 0:  # verificamos que el índice exista
                combined_sentences += sentences[j]['sentence'] + ''
            
        # Agregamos la oración actual
        combined_sentences += sentences[i]['sentence']
        
        # Agregamos las oraciones posteriores (según el buffer)
        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):  # verificamos que no se salga de la lista
                combined_sentences += '' + sentences[j]['sentence']

        # Guardamos el resultado en una nueva clave dentro del diccionario
        sentences[i]['combined_sentence'] = combined_sentences

    # Retornamos la lista actualizada
    return sentences


# Llamamos a la función para combinar oraciones con un buffer de 1 (una antes y una después)
sentences = combine_sentences(sentences)

from langchain.embeddings import OpenAIEmbeddings
oaiembeddings = OpenAIEmbeddings();

embeddings = oaiembeddings.embed_documents([x['combined_sentence'] for x in sentences])

for i, sentence in enumerate(sentences): 
    sentence['combined_sentence_embedding'] = embeddings[i]

from sklearn.metrics.pairwise import cosine_similarity
     
def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current: sentences[i]['']



