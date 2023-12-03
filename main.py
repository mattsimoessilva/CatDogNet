from neural_network import NeuralNetwork  # Supondo que o código da rede neural está em um arquivo chamado neural_network.py
from data_loader import load_data  # Supondo que você tenha uma função de carregamento de dados em um arquivo chamado data_loader.py

def main():
    # Carregar dados
    data_path = 'path/to/your/dataset'  # Substitua pelo caminho real
    X_train, Y_train, X_test = load_data(data_path)  # Carregue seus dados de treinamento e teste

    # Criar uma instância da rede neural
    n_x = len(X_train[0])
    n_h = 64  # Número de neurônios na camada oculta
    n_y = len(set(Y_train))  # Número de classes (assumindo classificação)

    neural_net = NeuralNetwork(n_x, n_h, n_y, alpha=0.01, batch_size=32, epochs=100, lambd=0.7)

    # Treinar a rede neural
    neural_net.train(X_train, Y_train)

    # Fazer previsões usando a rede neural treinada
    predictions = neural_net.predict(X_test)

    # Imprimir as previsões
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()

