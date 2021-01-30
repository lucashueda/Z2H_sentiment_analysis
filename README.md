# Zero2Hero em análise de sentimento em textos (Sentiment Analysis)

Você já sabe o básico de aprendizado de máquina? Já mexeu com as biblioecas Pandas, Numpy, Scikit-learn? Quer aprender como começar a mexer com textos?

Uma das tarefas mais simples envolvendo textos é a classificação de sentimentos. Dessa forma, vou consolidar nesse repositório 3 abordagens para se lidar com esse desafio. A primeira abordagem será a mais simples possível, a **Vetorização dos textos**, onde os textos são transformados em vetores e entçao podem ser utilizados como features para seu modelo. A segunda abordagem se baseia na ideia de transformar as palavras do texto em vetores densos, porém com propriedades **semânticas**, os famosos **Word Embeddings**. Por fim, vamos introduzir o conceito de atenção, onde não estamos mais preocupados em como representar as palavras, mas sim, selecionar quais delas devemos utilizar, nesse contexto, finalizaremos chegando nos mais recentes **Transformers**. Apenas o último tópico, exigirá algum conhecimento em redes neurais, porém o foco aqui não será ensinar sobre redes neurais (mas pretendo fazer um conteúdo sobre :D )

# Dados utilizados

Para todas as abordagens iremos utilizar a base de dados do IMDB (https://ai.stanford.edu/~amaas/data/sentiment/), que consistem em reviews de filmes classificados com um sentimento positivo ou negativo. É uma base bem simples, e ela permite que avaliemos nossos modelos nos mesmos conjuntos de dados, pois já vem com uma tag de treino e validação. 

# Estrutura do repositório

- 1 - Vectorizers: Pasta que consolida arquivos baseados na vetorização de textos
- 2 - Word Embeddings: Pasta que consolida arquivos baseados em word embeddings
- 3 - Attention: Pasta que consolida arquivos baseados em redes de atenção

IMPORTANTE: Para tornar mais direto a reproducibilidade cada etapa terá como principal código um Google Colab. Portanto, basta que você use o Collab para replicar os resultados apresentados aqui. Caso você queira utilizar localmente, a base de dados deverá estar em uma pasta "data" na root do repositório, e você deverá instalar os pacotes do requeriments.txt .


# Author

- Lucas Hideki Ueda, Mestrando em Deep Learning aplicado a conversão texto-fala (lucashueda@gmail.com)

# Referência (Em cada etapa terão referências específicas, aqui referenciaremos somente o que é exigido pelo uso da base de dados)

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts, "Learning Word Vectors for Sentiment Analysis", 2011. [https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf]
