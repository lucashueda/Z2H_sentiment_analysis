# Primeiro passo importar bibliotecas necessárias
import re # Para regex
import numpy as np # Para computação matricial
import pandas as pd # Para tratamento de dataframes
import collections # Para tratamentos de objetos de armazenamento

# Definindo a classe e implementando seus métodos
class CustomCountVectorizer:
  '''
    Classe que permite a vetorização de textos utilizando um tokenizador customizável e uma
    contagem de tokens com méétodo "fit" e "transform" para maior facilidade de tratamento de
    textos novos.
  '''


  def __init__(self, max_terms = 1.0, min_terms = 0.0, max_tokens = None, stopwords = None, numeric = False, lowercase = True):
    '''
      max_terms: float (0.0 - 1.0) que indica uma treshold de frequência máxima para fazer parte do vocabulário
      min_terms: float (0.0 - 1.0) que indica frequência mínima para fazer parte do vocabulário
      max_tokens: int (1 - len(vocab_)) que indica quantos dos "N" tokens mais frequentes do vocabulário serão considerados
      stopwords: lista de str com as palavras a serem retiradas do vocabulário
      numeric: boolean que indica se será considerado números como tokens (True) ou não (False)
      lowercase: boolean que indica se é desejado transformar todos os caracteres do texto para minúsculo (True) ou não (False)
    '''

    # Parâmetros locais do objeto instanciado
    self.vocab_ = collections.defaultdict(int) # Um dict que retorna 0 (por conta do argumento "int") caso a key passada nunca tenha sido definida
    #self.vocab_ = dict()
    self.stopwords_ = stopwords # O vetor de stopwords que seŕa armazenado como parâmetro do objeto instanciado
    
    # Parâmetro do algoritmo
    self.max_terms = max_terms
    self.min_terms = min_terms
    self.max_tokens = max_tokens
    self.numeric = numeric
    self.lowercase = lowercase

  # Função que retira caracteres especiais e pontuações dos textos e retorna tokens limpos de stopwords e em lowercase caso desejado
  def preprocess(self, text):
    '''
      text: array de chars
      stopwords: lista de palavras (str) que serão removidas
      lowercase: parâmetro se transformará em lowercase
      numeric: parâmetro se considera números tokens ou não
    '''

    # Primeiro passo tirar caracteres especiais e pontuações com regex
    result = re.sub('[^A-Za-z^0-9]+',' ', text)

    # Caso numeric seja falso tiramos também os números
    if(self.numeric):
      result = re.sub(r'(\s\d+)', '', result)

    # Caso lowercase seja True jogamos tudo para lowercase
    if(self.lowercase):
      result = result.lower()
    
    return result

  # Função que pega um texto e gera tokens do mesmo
  def tokenizer(self, text):

    # Damos o split para transformar a string em um vetor de tokens
    tokens = text.split()

    # Limpamos as stopwords caso seja uma lista e não vazia
    if((self.stopwords_ != None) & (type(self.stopwords_) == list)):
      tokens_no_sw = [w for w in tokens if w not in self.stopwords_]
      tokens = tokens_no_sw
    return tokens

  # Função gera o vocabulário do objeto instanciado
  def fit_transform(self, vet_text):

    # Nossa mochila de tokens, bag of tokens (bot)
    bot = []

    # Iteramos por cada texto do vetor de textos "vet_text"
    for tx in vet_text:
      # Para cada texto "tx" nóós preprocessamos ele
      actual_text = self.preprocess(tx)
      # Depois tokenizamos ele
      actual_tokens = self.tokenizer(actual_text)
      # Depois adicionamos na nossa bag of tokens
      bot.extend(actual_tokens)

    # Finalizado a etapa de gerar todos os tokens dos textos passados, já processados, vamos gerar o vocab_
    for t in bot:
      self.vocab_[t] = self.vocab_.get(t, 0) + 1/len(bot)

    # Limita o vocabulário conforme os argumentos passados ao objeto
    if((self.max_terms < 1.0)|(self.min_terms > 0.0)):
      # Criando dict auxiliar com keys filtradas
      filter_vocab = dict((k,v) for k, v in self.vocab_.items() if v > self.min_terms and v < self.max_terms)

      # Retornando os valores para nosso vocabulário
      self.vocab_ = collections.defaultdict(int, filter_vocab)

    # Caso self.max_tokens seja diferente de None então pegaremos as max_tokens mais frequentes tokens para o vocab
    if(self.max_tokens != None):
      self.vocab_ = collections.defaultdict(int,sorted(self.vocab_.items(), key = lambda x: x[1], reverse = True)[:self.max_tokens])

    # Parâmetro interno que é uma tupla dos tokens do vocab_ ordenados em ordem descrescente
    self.sorted_vocab_ = sorted(self.vocab_.items(), key = lambda x: x[1], reverse = True)

    # Transformando o vocab_ agora em um dict de palavras e índices para construirmos a matriz 
    self.vocab_ = dict((k,v) for v, k in enumerate(self.vocab_.keys()))
    
    # Finalmente definindo a matriz de vetorização
    self.map_matrix = np.zeros((len(vet_text),len(self.vocab_)))

    # Percorrendo os texto e aplicando o count de tokens do vocabulário
    for i, text in enumerate(vet_text):
      # Para cada texto "tx" nós preprocessamos ele
      actual_text = self.preprocess(text)
      # Depois tokenizamos ele
      actual_tokens = self.tokenizer(actual_text)

      for tk in actual_tokens:
        if(tk in self.vocab_.keys()):
          self.map_matrix[i][self.vocab_[tk]] += 1
    return self.map_matrix