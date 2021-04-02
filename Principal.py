import  nltk
from nltk.metrics import ConfusionMatrix
import BasesFrases.Base_treinamento
import BasesFrases.Base_teste
import BasesFrases.Stop_words
import Stemmer
import Erros_classificador

# baixar atualizacoes
#nltk.download()

# variaveis
baseTreinamento = BasesFrases.Base_treinamento.vet_baseTreinamento
baseTeste = BasesFrases.Base_teste.vet_baseTeste
stopWords = BasesFrases.Stop_words.stopWordsNLTK

# aplicando stemming
frasesComStemmingTreinamento = Stemmer.aplicaStemmer(baseTreinamento)
frasesComStemmingTeste = Stemmer.aplicaStemmer(baseTeste)

# busca cada uma das palavras, apos quebrar todas em radicais
palavrasTreinamento = Stemmer.buscaPalavras(frasesComStemmingTreinamento)
palavrasTeste = Stemmer.buscaPalavras(frasesComStemmingTeste)

# quantidade de vezes que uma palavra se repete
frequenciaTreinamento = Stemmer.buscaFrequencia(palavrasTreinamento)
frequenciaTeste = Stemmer.buscaFrequencia(palavrasTeste)

# palavras que nao se repetem
palavrasUnicasTreinamento = Stemmer.buscaPalavrasUnicas(frequenciaTreinamento)
palavrasUnicasTeste = Stemmer.buscaPalavrasUnicas(frequenciaTeste)

# extracao das palavras de uma frase que foram passada por parametro
def extratorPalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasUnicasTreinamento:
        # para percorrer as palavras de uma frase
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

# passa como parametro radicais de palavras
# mostra se existe a palavra na frase ou nao, com tru ou false
# extrai as que estao sendo passadas por paramento
caracteristicasFrase = extratorPalavras(['am', 'nov', 'dia'])

# criacao das bases completas ja modificadas
# passa como parametros a funcao que extrai as palavras e a var frasescomstemming, pq a base ja esta processada
# essa funcao apply_features, ja faz o processamento necessario para toda a base
baseCompletaTreinamento = nltk.classify.apply_features(extratorPalavras, frasesComStemmingTreinamento)
baseCompletaTeste = nltk.classify.apply_features(extratorPalavras, frasesComStemmingTeste)

# constroi a tabela de probabilidade
# o classificador verifica quais palavras estao relacionada a quais emocoes, e assim eh possivel saber se o algoritmo esta acertando ou nao
classificador = nltk.NaiveBayesClassifier.train(baseCompletaTreinamento)
# imprime as labels (emocoes) da base usada
#print(classificador.labels())
# imprime os dados mais informativos, com maior probabilidade de ser certa emocao
#print(classificador.show_most_informative_features(20))

# mostra a precisao do algoritmo (em porcentagem) usando o accuracy do nltk, com base na tabela de probabilidades criada com naivebayes
# eh necessario prestar atencao a porcentagem de precisao, ideal eh acima de 70%
#print(nltk.classify.accuracy(classificador, basecompletatreinamento)) # 93% pq usa a mesma base para a comparacao
#print(nltk.classify.accuracy(classificador, baseCompletaTeste)) # 32% pq usa bases diferentes (EH A FORMA CERTA DE VERIFICAR)

# exibe onde o classificador errou
# o primeiro parametro, mostra qual eh a classificacao correta
# o segunto mostra qual o programa disse que eh
# o terceiro mostra as palavras que estao sendo confundidas com frequencia
#Erros_classificador.erros_classificador(classificador, baseCompletaTeste)

# o numero que estiver dentro de <> sao os que o programa acertou, os demais numeros foram erros e qual a categoria escolhida
esperado = []
previsto = []

# montagem das emocoes esperadas e previstas da base completa
for (frase, classe) in baseCompletaTeste:
        resultado = classificador.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)

# a matriz faz a juncao das duas variaveis, esperado e previsto
matriz = ConfusionMatrix(esperado, previsto)

# imprime a matriz de confusao
#print(matriz)

frasePersonalizada = input("Digite alguma frase: ")

while(frasePersonalizada != "sair"):
    testeStemming = []
    stemmer = nltk.stem.RSLPStemmer()
    # criando stemmer da frase personalizada (frasePersonalizada)
    # extrair radicais
    for (palavrasTreinamento) in frasePersonalizada.split():
        comStem = [p for p in palavrasTreinamento.split()]
        testeStemming.append(str(stemmer.stem(comStem[0])))

    #print(testeStemming)

    # lista de trues e falses onde a frase personalizada se encaixa
    novo = extratorPalavras(testeStemming)
    #print(novo)

    # classicaficacao da frase
    print('Frase: ' + frasePersonalizada)
    print('Identificamos que essa frase é de : ' + classificador.classify(novo))
    print()

    # mostra a probabilidade, porcentagem de cada emocao
    distribuicao = classificador.prob_classify(novo)
    #for classe in distribuicao.samples():
        #print("%s: %f" % (classe, distribuicao.prob(classe)))

    frasePersonalizada = input("Digite alguma frase: ")