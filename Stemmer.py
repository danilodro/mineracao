import nltk
import BasesFrases.Stop_words

__stopWords = BasesFrases.Stop_words.stopWordsNLTK # dois underlines eh private

# funcao para pegar o radical das palavras
# (livr)    o
# (livr)    inho
def aplicaStemmer(texto):
    stemmer = nltk.stem.RSLPStemmer() # RSLP eh para o portugues
    frasesStemming = []
    for (palavras, emocao) in texto:
        # ja pega o radical das palavras, verificando se nao estao na lista de stopwords
        comStemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in __stopWords]
        # adiciona a lista de frasesStemming
        frasesStemming.append((comStemming, emocao))
    return frasesStemming

# funcao para pegar todas as palavras apos o stemming, sem a emocao
def buscaPalavras(frases):
    todasPalavras = []
    for (palavras, emocao) in frases:
        # pega somente as palavras
        todasPalavras.extend(palavras)
    return  todasPalavras

# busca quantas vezes uma palavra de repete
def buscaFrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

# retorna a lista das palavras unicas
def buscaPalavrasUnicas(frequencia):
    # pega so os valores
    freq = frequencia.keys()
    return freq
