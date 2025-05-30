# 🧠 Perceptron para Funções Lógicas

Implementação do algoritmo **Perceptron** para resolver funções lógicas **AND**, **OR** e demonstrar sua limitação com **XOR**, com suporte para múltiplas entradas booleanas. Este projeto tem caráter educacional e visa ilustrar os fundamentos de aprendizado supervisionado com redes neurais lineares.

---

## 📋 Índice

- [📖 Descrição](#-descrição)
- [✨ Características](#-características)
- [🚀 Instalação](#-instalação)
- [💻 Como Usar](#-como-usar)
- [📊 Resultados](#-resultados)
- [🧠 Análise Teórica](#-análise-teórica)
- [📝 Licença](#-licença)

---

## 📖 Descrição

O **Perceptron** foi proposto por **Frank Rosenblatt** em 1957 e é um modelo simples de neurônio artificial capaz de resolver problemas de classificação binária **linearmente separáveis**.

Este projeto demonstra:

- Como o Perceptron resolve problemas lineares (AND e OR)
- A evolução dos **hiperplanos de separação**
- A **falha com a função XOR**, que não é linearmente separável
- A evolução dos **pesos e bias** durante o treinamento

---

## ✨ Características

- ✅ Suporte para **n entradas booleanas** (2 a 10)
- ✅ Escolha da função lógica (**AND**, **OR**, **XOR**)
- ✅ Visualização gráfica para problemas com 2 entradas
- ✅ Evolução dos **pesos**, **bias** e **acurácia**
- ✅ Animação do **hiperplano de decisão** (para 2D)
- ✅ Interface interativa no terminal

---

## 🚀 Instalação

### ✅ Pré-requisitos

- Python 3.7 ou superior

### 📦 Instale as dependências:

```bash
pip install numpy matplotlib
````

### 🔄 Clonando o repositório:

git clone https://github.com/sabarense/Perceptron.git<br>
cd Perceptron<br>
pip install -r requirements.txt 

### 💻 Como Usar

```bash
python perceptron.py
```
O terminal solicitará:

Qual função lógica deseja treinar (AND, OR, XOR)

Quantidade de entradas (mínimo 2, máximo 10)

Se forem escolhidas 2 entradas, será gerado um gráfico animado mostrando a evolução do hiperplano de separação.

### 📊 Resultados

🔹 Função AND (2 entradas)

    Convergência rápida

    Visualização clara do plano de separação

    Pesos se estabilizam rapidamente

🔹 Função OR (4 entradas)

    Classificação correta mesmo com mais dimensões
    
    Apenas saída textual (sem gráfico para >2D)

🔹 Função XOR

    Exibe as limitações do Perceptron
    
    Não converge, pois os dados não são linearmente separáveis
    
    Gráfico mostra falha do hiperplano

## 🧠 Análise Teórica
O Perceptron é um classificador linear binário. Ele aprende ajustando os pesos com base em erros, usando a regra:<br>

````
wᵢ ← wᵢ + η * (y - ŷ) * xᵢ

Onde:

η é a taxa de aprendizado

y é a saída esperada

ŷ é a saída predita

xᵢ é a i-ésima entrada
````
### ❗ Limitação com XOR

O modelo não consegue aprender funções não linearmente separáveis, como o XOR. Isso ilustra a necessidade de arquiteturas mais complexas, como Redes Neurais Multicamadas (MLP).

### 📝 Licença
Este projeto é de uso educacional e está sob a licença MIT.
Sinta-se à vontade para estudar, modificar e reutilizar o código.


