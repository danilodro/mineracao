import nltk

# lista de palavras que nao tem um significado na frase (para sentimento)
# STOPWORDS MANUAL
stopWordsNLTK = nltk.corpus.stopwords.words('portuguese')
stopWordsNLTK.append('vou')
stopWordsNLTK.append('tão')

# funcao para remocao de stopwords
def removeStopwords(texto):
    frases = []
    # 'palavras' pega cada palavra da frase
    # 'emocao' pega a emocao no final da frase
    for (palavras, emocao) in texto:
        # quebra a frase em palavras e pega somente as que nao estiverem dentro de stopwords
        semStop = [p for p in palavras.split() if p not in stopWordsNLTK]
        # adiciona a nova frase quebrada com a sua respectiva emocao (aprendizagem supervisionada)
        frases.append((semStop, emocao))

    # retorna a frase sem as stop words
    return frases