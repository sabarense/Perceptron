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
        return 1 if x >= 0 else 0

    def _predict_single(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return self._activation_function(linear_output)

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape
        self.weights = np.random.uniform(-0.5, 0.5, n_features)
        self.bias = np.random.uniform(-0.5, 0.5)
        self.training_history = []

        for epoch in range(self.max_epochs):
            errors = 0
            epoch_weights = self.weights.copy()
            epoch_bias = self.bias

            for x_i, target in zip(X, y):
                prediction = self._predict_single(x_i)
                error = target - prediction

                if error != 0:
                    errors += 1
                    self.weights += self.learning_rate * error * x_i
                    self.bias += self.learning_rate * error

            self.training_history.append({
                'epoch': epoch,
                'weights': epoch_weights,
                'bias': epoch_bias,
                'errors': errors
            })

            if verbose and epoch % 100 == 0:
                print(f"Época {epoch}: {errors} erros")

            if errors == 0:
                if verbose:
                    print(f"Convergência alcançada na época {epoch}")
                break

        return self


def generate_logic_data(n_inputs, logic_function):
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


def plot_training_evolution(perceptron, X, y, title):
    total_epochs = len(perceptron.training_history)
    show_epochs = [0, total_epochs // 4, total_epochs // 2, total_epochs - 1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for idx, epoch in enumerate(show_epochs[:4]):
        if epoch < len(perceptron.training_history):
            ax = axes[idx]
            hist = perceptron.training_history[epoch]
            weights = hist['weights']
            bias = hist['bias']

            colors = ['red', 'blue']
            for class_value in [0, 1]:
                mask = y == class_value
                ax.scatter(X[mask, 0], X[mask, 1], c=colors[class_value],
                           s=100, alpha=0.7, label=f"Classe {class_value}")

            if len(weights) >= 2 and abs(weights[1]) > 1e-6:
                x1_range = np.linspace(-0.5, 1.5, 100)
                x2_range = -(weights[0] * x1_range + bias) / weights[1]
                ax.plot(x1_range, x2_range, 'g--', linewidth=2,
                        label='Hiperplano de Separação')

            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 1.5)
            ax.set_title(f"{title} - Época {epoch} - Erros: {hist['errors']}")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_decision_boundary_2d(perceptron, X, y, title):
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue']

    for class_value in [0, 1]:
        mask = y == class_value
        plt.scatter(X[mask, 0], X[mask, 1],
                    c=colors[class_value], label=f"Classe {class_value}",
                    s=100, alpha=0.7)

    if perceptron.weights is not None and len(perceptron.weights) >= 2:
        w1, w2 = perceptron.weights[0], perceptron.weights[1]
        bias = perceptron.bias

        if abs(w2) > 1e-6:
            x1_range = np.linspace(-0.5, 1.5, 100)
            x2_range = -(w1 * x1_range + bias) / w2
            plt.plot(x1_range, x2_range, 'g--', linewidth=2,
                     label='Fronteira de Decisão')

    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.title(title)
    plt.xlabel('Entrada 1')
    plt.ylabel('Entrada 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def demonstrate_xor_limitation():
    print("\n=== TESTE: LIMITAÇÃO DO PERCEPTRON COM XOR ===")
    X, y = generate_logic_data(2, 'XOR')

    perceptron = Perceptron()
    perceptron.fit(X, y, verbose=True)

    predictions = perceptron.predict(X)
    accuracy = np.mean(predictions == y)

    print("\nResultados para XOR:")
    for xi, yi, pi in zip(X, y, predictions):
        print(f"Entrada: {xi} -> Predito: {pi} | Esperado: {yi}")
    print(f"Acurácia: {accuracy:.2%}")
    plot_training_evolution(perceptron, X, y, "Treinamento XOR (falha)")
    plot_decision_boundary_2d(perceptron, X, y, "Tentativa de Classificação XOR")


def main():
    print("=== ALGORITMO PERCEPTRON - Funções Lógicas ===")
    print("Funções disponíveis: AND, OR, XOR")

    logic_function = input("Escolha a função lógica (AND / OR / XOR): ").strip().upper()
    if logic_function not in ['AND', 'OR', 'XOR']:
        print("Função inválida. Escolha entre AND, OR ou XOR.")
        return

    try:
        n_inputs = int(input("Número de entradas (>=2): "))
        if n_inputs < 2:
            raise ValueError
    except ValueError:
        print("Número inválido de entradas.")
        return

    if logic_function == 'XOR' and n_inputs != 2:
        print("XOR só é suportado com 2 entradas.")
        return

    X, y = generate_logic_data(n_inputs, logic_function)
    perceptron = Perceptron(learning_rate=0.1, max_epochs=1000)
    perceptron.fit(X, y, verbose=True)

    predictions = perceptron.predict(X)
    accuracy = np.mean(predictions == y)

    print(f"\n--- RESULTADOS PARA {logic_function} ({n_inputs} entradas) ---")
    print(f"Acurácia: {accuracy:.2%}")
    for xi, yi, pi in zip(X, y, predictions):
        print(f"{xi} -> Predito: {pi} | Esperado: {yi}")

    if n_inputs == 2:
        plot_training_evolution(perceptron, X, y, f"Evolução - {logic_function}")
        plot_decision_boundary_2d(perceptron, X, y, f"Perceptron - {logic_function}")

    if logic_function == 'XOR':
        print("\n⚠️ O Perceptron não consegue resolver problemas não linearmente separáveis como o XOR.")


if __name__ == "__main__":
    main()
