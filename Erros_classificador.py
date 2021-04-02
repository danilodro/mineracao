def erros_classificador(classificador, baseCompletaTeste):
    # verificar onde estao os erros
    # percorre todos os registros
    erros = []

    for (frase, classe) in baseCompletaTeste:
        print(frase)
        print(classe)

        resultado = classificador.classify(frase)

        #  se o resultado for diferente, adiciona os parametros ao erro
        if resultado != classe:
            erros.append((classe, resultado, frase))

    # mostra todos os erros que o classificador cometeu
    for (classe, resultado, frase) in erros:
        # o primeiro parametro, mostra qual eh a classificacao correta
        # o segunto mostra qual o programa disse que eh
        # o terceiro mostra as palavras que estao sendo confundidas com frequencia
        print(classe, resultado, frase)