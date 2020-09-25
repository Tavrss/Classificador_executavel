import cv2 as cv
import numpy as np
import pandas as pd


# função mostrar para quando precisarmos ver a imgem
def mostrar(imagem):
    cv.namedWindow('janela', cv.WINDOW_NORMAL)
    cv.imshow('janela', imagem)
    cv.waitKey()
    cv.destroyAllWindows()


# feita para salvar os dados em formato de um dataframe
def criando_dataframe(first_name, concentrations, timesvar,
                      first_channelN, second_channelN, third_channelN,
                      first_channelM, second_channelM, third_channelM,
                      first_channelVar, second_channelVar, third_channelVar,
                      formatN):
    dados = pd.DataFrame(
        {'Image': first_name, 'Concentration': concentrations, 'Time': timesvar,
         # primeira dimensão
         f'{first_channelN} M': first_channelM,
         f'{first_channelN} Var': first_channelVar,
         # segunda dimensão
         f'{second_channelN} M': second_channelM,
         f'{second_channelN} Var': second_channelVar,
         # terceira dimensão
         f'{third_channelN} M': third_channelM,
         f'{third_channelN} Var': third_channelVar})
    return dados


# pega o nome e o separa para colocar no dataframe
def pega_nome(nome_da_imgem):
    concentration = int(nome_da_imgem[0:4])
    timevar = int(nome_da_imgem[3:6])
    nome_da_imgem = nome_da_imgem.split('.')
    first_nome = nome_da_imgem[0]
    return first_nome, concentration, timevar


'''
##############################################Segmentação######################################################
'''

# Vetores que guardarão as informações
# primeira dimensão
# média
primeiraDMean = [[], [], [], []]
# variância
primeiraDVar = [[], [], [], []]
# segunda dimensão
# média
segundaDMean = [[], [], [], []]
# variância
segundaDVar = [[], [], [], []]
# terceira simensão
# média
terceiraDMean = [[], [], [], []]
# variância
terceiraDVar = [[], [], [], []]
# variaveis para nomenclatura
primeiroD = ['Luminosity', 'Hue', 'Y', 'Blue']
segundoD = ['Chromatic Coordinate a', 'Saturation', 'Cr', 'Green']
terceiroD = ['Chromatic Coordinate b', 'Value', 'Cb', 'Red']
nomes_dos_formatos = ['Lab', 'HSV', 'YCrCb', 'RGB']
dataframe_da_imgem = [[], [], [], []]  # dataframe a ser classificado
kernel = np.ones((13, 13), np.uint8)  # necessário para filtro morfológico

nome = input('Digite o nome da imagem contida no diretório: ')
img = cv.imread(nome, 1)  # abrindo a imagem
img = img[1200:2500, 1300:3200]  # recote padrão já pré definido
img2 = cv.cvtColor(img, cv.COLOR_BGR2Lab)
(L, Ca, Cb) = cv.split(img2)  # dividir canal de cores
Cb = cv.blur(Cb, (11, 11))  # blur
opening = cv.morphologyEx(Cb, cv.MORPH_CLOSE, kernel)  # filtro morfológico
t_usado, t_img = cv.threshold(opening, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # otsu aplicado na imagem
borda = cv.Canny(t_img, 100, 200)
imgcon, contors, hiera = cv.findContours(borda, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # seleção dos contornos
cnt = contors[0]  # contor da primenira posição da hierarquia
x, y, w, h = cv.boundingRect(cnt)
cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)  # retangulo aproximado a partir dos contornos
roi = img[y + 35:y + h - 35, x + 35:x + w - 35]  # região de interesse

'''
#####################################Extração de caracteristica################################################
'''

(primeiro_nome, concentracao, tempo) = pega_nome(nome)  # recebe os nomes
for formatos in range(4):
    # os valores de média e variancia de cada canal de cor é calculado e salvo em um vetor
    if formatos == 0:
        dimensoes = cv.cvtColor(roi, cv.COLOR_BGR2Lab)  # os flags daqui tem que ser BGR2algo
        (primeiro, segundo, terceiro) = cv.split(dimensoes)
        # média
        primeiraDMean[formatos].append(np.mean(primeiro))
        segundaDMean[formatos].append(np.mean(segundo))
        terceiraDMean[formatos].append(np.mean(terceiro))
        # variância
        primeiraDVar[formatos].append(np.var(primeiro))
        segundaDVar[formatos].append(np.var(segundo))
        terceiraDVar[formatos].append(np.var(terceiro))
    elif formatos == 1:
        dimensoes = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        (primeiro, segundo, terceiro) = cv.split(dimensoes)
        # média
        primeiraDMean[formatos].append(np.mean(primeiro))
        segundaDMean[formatos].append(np.mean(segundo))
        terceiraDMean[formatos].append(np.mean(terceiro))
        # variância
        primeiraDVar[formatos].append(np.var(primeiro))
        segundaDVar[formatos].append(np.var(segundo))
        terceiraDVar[formatos].append(np.var(terceiro))
    elif formatos == 2:
        dimensoes = cv.cvtColor(roi, cv.COLOR_BGR2YCrCb)
        (primeiro, segundo, terceiro) = cv.split(dimensoes)
        # média
        primeiraDMean[formatos].append(np.mean(primeiro))
        segundaDMean[formatos].append(np.mean(segundo))
        terceiraDMean[formatos].append(np.mean(terceiro))
        # variância
        primeiraDVar[formatos].append(np.var(primeiro))
        segundaDVar[formatos].append(np.var(segundo))
        terceiraDVar[formatos].append(np.var(terceiro))
    else:
        dimensoes = roi
        (primeiro, segundo, terceiro) = cv.split(dimensoes)
        # média
        primeiraDMean[formatos].append(np.mean(primeiro))
        segundaDMean[formatos].append(np.mean(segundo))
        terceiraDMean[formatos].append(np.mean(terceiro))
        # variância
        primeiraDVar[formatos].append(np.var(primeiro))
        segundaDVar[formatos].append(np.var(segundo))
        terceiraDVar[formatos].append(np.var(terceiro))
# salvando dados em um dataframe
for varnum in range(4):
    dataframe_da_imgem[varnum] = criando_dataframe(primeiro_nome, concentracao, tempo,
                                                   primeiroD[varnum], segundoD[varnum], terceiroD[varnum],
                                                   primeiraDMean[varnum], segundaDMean[varnum], terceiraDMean[varnum],
                                                   primeiraDVar[varnum], segundaDVar[varnum], terceiraDVar[varnum],
                                                   nomes_dos_formatos[varnum])

vetor = pd.concat(dataframe_da_imgem, axis=1)

'''
o resultado final é um vetor de 4 posições nas quais os valore de cada sistema de cor é salvo.
'''

'''
tentativa de criação de um executavel para classificação automatica de fitas colorimétricas.
'''
