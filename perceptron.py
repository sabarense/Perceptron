import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class Perceptron:
    def __init__(self, learning_rate=0.1, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.training_history = []

    def _activation_function(self, x):
        """Função de ativação step (Heaviside)"""
        return 1 if x >= 0 else 0

    def _predict_single(self, x):
        """Predição para uma única amostra"""
        linear_output = np.dot(x, self.weights) + self.bias
        return self._activation_function(linear_output)

    def predict(self, X):
        """Predição para múltiplas amostras"""
        return [self._predict_single(x) for x in X]

    def fit(self, X, y, verbose=False):
        """Treinamento do Perceptron"""
        n_samples, n_features = X.shape

        # Inicialização dos pesos e bias
        self.weights = np.random.uniform(-0.5, 0.5, n_features)
        self.bias = np.random.uniform(-0.5, 0.5)

        self.training_history = []

        for epoch in range(self.max_epochs):
            errors = 0
            epoch_weights = self.weights.copy()
            epoch_bias = self.bias

            for i, (x_i, target) in enumerate(zip(X, y)):
                prediction = self._predict_single(x_i)
                error = target - prediction

                if error != 0:
                    errors += 1
                    # Atualização dos pesos usando a regra do Perceptron
                    self.weights += self.learning_rate * error * x_i
                    self.bias += self.learning_rate * error

            # Salvar histórico para visualização
            self.training_history.append({
                'epoch': epoch,
                'weights': epoch_weights,
                'bias': epoch_bias,
                'errors': errors
            })

            if verbose and epoch % 100 == 0:
                print(f"Época {epoch}: {errors} erros")

            # Convergência alcançada
            if errors == 0:
                if verbose:
                    print(f"Convergência alcançada na época {epoch}")
                break

        return self


def generate_logic_data(n_inputs, logic_function):
    """Gera dados para funções lógicas com n entradas"""
    # Gerar todas as combinações possíveis de 0s e 1s
    X = np.array(list(product([0, 1], repeat=n_inputs)))

    if logic_function == 'AND':
        y = np.array([1 if all(row) else 0 for row in X])
    elif logic_function == 'OR':
        y = np.array([1 if any(row) else 0 for row in X])
    elif logic_function == 'XOR':
        y = np.array([1 if sum(row) % 2 == 1 else 0 for row in X])
    else:
        raise ValueError("Função lógica não suportada")

    return X, y


def plot_training_evolution(perceptron, X, y, title, show_epochs=None):
    """Plota a evolução do hiperplano durante o treinamento"""
    if show_epochs is None:
        # Selecionar épocas automaticamente
        total_epochs = len(perceptron.training_history)
        if total_epochs > 4:
            show_epochs = [0, total_epochs // 4, total_epochs // 2, total_epochs - 1]
        else:
            show_epochs = list(range(total_epochs))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for idx, epoch in enumerate(show_epochs[:4]):
        if epoch < len(perceptron.training_history):
            ax = axes[idx]

            # Dados históricos da época
            hist = perceptron.training_history[epoch]
            weights = hist['weights']
            bias = hist['bias']

            # Plotar pontos
            colors = ['red', 'blue']
            labels = ['Classe 0', 'Classe 1']
            for class_value in [0, 1]:
                mask = y == class_value
                ax.scatter(X[mask, 0], X[mask, 1], c=colors[class_value],
                           s=100, alpha=0.7, label=labels[class_value])

            # Plotar hiperplano da época
            if len(weights) >= 2 and abs(weights[1]) > 1e-6:
                x1_range = np.linspace(-0.5, 1.5, 100)
                x2_range = -(weights[0] * x1_range + bias) / weights[1]
                ax.plot(x1_range, x2_range, 'g--', linewidth=2,
                        label='Hiperplano de Separação')

            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 1.5)
            ax.set_xlabel('Entrada 1')
            ax.set_ylabel('Entrada 2')
            ax.set_title(f'{title} - Época {epoch}\nErros: {hist["errors"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_decision_boundary_2d(perceptron, X, y, title, epoch=None):
    """Plota a fronteira de decisão para problemas 2D"""
    plt.figure(figsize=(10, 6))

    # Plotar pontos de dados
    colors = ['red', 'blue']
    labels = ['Classe 0', 'Classe 1']

    for class_value in [0, 1]:
        mask = y == class_value
        plt.scatter(X[mask, 0], X[mask, 1],
                    c=colors[class_value], label=labels[class_value],
                    s=100, alpha=0.7)

    # Plotar linha de decisão se os pesos foram treinados
    if perceptron.weights is not None and len(perceptron.weights) >= 2:
        w1, w2 = perceptron.weights[0], perceptron.weights[1]
        bias = perceptron.bias

        # Linha de decisão: w1*x1 + w2*x2 + bias = 0
        # x2 = -(w1*x1 + bias) / w2
        if abs(w2) > 1e-6:  # Evitar divisão por zero
            x1_range = np.linspace(-0.5, 1.5, 100)
            x2_range = -(w1 * x1_range + bias) / w2
            plt.plot(x1_range, x2_range, 'g--', linewidth=2,
                     label='Fronteira de Decisão')

    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('Entrada 1')
    plt.ylabel('Entrada 2')
    plt.title(title if epoch is None else f"{title} - Época {epoch}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def test_perceptron_logic_gates():
    """Testa o Perceptron nas funções lógicas"""

    print("=== TESTE DO PERCEPTRON ===\n")

    # Teste com diferentes números de entradas
    for n_inputs in [2, 3, 4]:
        print(f"\n--- Testando com {n_inputs} entradas ---")

        for logic_func in ['AND', 'OR']:
            print(f"\nFunção {logic_func}:")

            # Gerar dados
            X, y = generate_logic_data(n_inputs, logic_func)

            # Treinar Perceptron
            perceptron = Perceptron(learning_rate=0.1, max_epochs=1000)
            perceptron.fit(X, y, verbose=True)

            # Testar predições
            predictions = perceptron.predict(X)
            accuracy = np.mean(predictions == y)

            print(f"Acurácia: {accuracy:.2%}")
            print(f"Pesos finais: {perceptron.weights}")
            print(f"Bias final: {perceptron.bias:.3f}")

            # Mostrar algumas predições
            print("Exemplos de predições:")
            for i in range(min(8, len(X))):
                print(f"  {X[i]} -> {predictions[i]} (esperado: {y[i]})")

            # Plotar para casos 2D
            if n_inputs == 2:
                # Mostrar evolução do treinamento
                plot_training_evolution(perceptron, X, y, f"Evolução - {logic_func}")
                # Mostrar resultado final
                plot_decision_boundary_2d(perceptron, X, y, f"Perceptron - {logic_func}")


def demonstrate_xor_limitation():
    """Demonstra que o Perceptron não resolve XOR"""
    print("\n=== DEMONSTRAÇÃO: LIMITAÇÃO DO PERCEPTRON COM XOR ===\n")

    X, y = generate_logic_data(2, 'XOR')

    print("Dados XOR:")
    for i in range(len(X)):
        print(f"  {X[i]} -> {y[i]}")

    # Tentar treinar Perceptron para XOR
    perceptron = Perceptron(learning_rate=0.1, max_epochs=1000)
    perceptron.fit(X, y, verbose=True)

    predictions = perceptron.predict(X)
    accuracy = np.mean(predictions == y)

    print(f"\nResultados do treinamento XOR:")
    print(f"Acurácia: {accuracy:.2%}")
    print(f"Pesos finais: {perceptron.weights}")
    print(f"Bias final: {perceptron.bias:.3f}")

    print("\nPredições vs Esperado:")
    for i in range(len(X)):
        print(f"  {X[i]} -> {predictions[i]} (esperado: {y[i]})")

    # Mostrar evolução do treinamento (tentativa de aprender XOR)
    plot_training_evolution(perceptron, X, y, "Evolução XOR (Falha)")

    # Plotar resultado final
    plot_decision_boundary_2d(perceptron, X, y, "Perceptron - XOR (Falha)")

    print("\nEXPLICAÇÃO:")
    print("O XOR não é linearmente separável. Não existe uma única linha reta")
    print("que possa separar as classes 0 e 1 no espaço bidimensional.")
    print("Por isso o Perceptron simples falha em aprender esta função.")


def interactive_perceptron():
    """Função interativa para o usuário escolher parâmetros"""
    print("\n=== PERCEPTRON INTERATIVO ===")

    # Solicitar entrada do usuário
    while True:
        try:
            n_inputs = int(input("\nDigite o número de entradas (2-10): "))
            if 2 <= n_inputs <= 10:
                break
            else:
                print("Por favor, digite um número entre 2 e 10.")
        except ValueError:
            print("Por favor, digite um número válido.")

    print("\nFunções disponíveis:")
    print("1. AND")
    print("2. OR")
    print("3. XOR (para demonstrar limitação)")

    while True:
        try:
            choice = int(input("Escolha a função (1, 2 ou 3): "))
            if choice in [1, 2, 3]:
                logic_functions = {1: 'AND', 2: 'OR', 3: 'XOR'}
                logic_func = logic_functions[choice]
                break
            else:
                print("Por favor, escolha 1, 2 ou 3.")
        except ValueError:
            print("Por favor, digite um número válido.")

    # Gerar dados e treinar
    X, y = generate_logic_data(n_inputs, logic_func)
    perceptron = Perceptron(learning_rate=0.1, max_epochs=1000)

    print(f"\nTreinando Perceptron para {logic_func} com {n_inputs} entradas...")
    perceptron.fit(X, y, verbose=True)

    # Resultados
    predictions = perceptron.predict(X)
    accuracy = np.mean(predictions == y)

    print(f"\nResultados:")
    print(f"Acurácia: {accuracy:.2%}")
    print(f"Pesos finais: {perceptron.weights}")
    print(f"Bias final: {perceptron.bias:.3f}")

    # Mostrar todas as predições
    print(f"\nTabela verdade completa para {logic_func}:")
    print("Entradas -> Predição (Esperado)")
    for i in range(len(X)):
        entrada_str = str(X[i]).replace('[', '').replace(']', '')
        print(f"{entrada_str} -> {predictions[i]} ({y[i]})")

    # Plotar se for 2D
    if n_inputs == 2:
        plot_training_evolution(perceptron, X, y, f"Evolução - {logic_func}")
        plot_decision_boundary_2d(perceptron, X, y, f"Resultado Final - {logic_func}")


if __name__ == "__main__":
    # Executar testes automáticos
    test_perceptron_logic_gates()
    demonstrate_xor_limitation()

    # Executar modo interativo
    interactive_perceptron()
