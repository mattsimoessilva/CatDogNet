# CatDogNet: A Binary Image Classifier

CatDogNet is a binary image classifier implemented in mostly pure Python. The project aims to classify images of cats and dogs using a neural network built from scratch.

## Project Structure

The project is organized into several components:

- `image_processing.py`: Contains functions for loading, resizing, and normalizing images, creating labels based on filenames, and splitting data into training and testing sets.
- `math_functions.py`: Contains implementations of basic mathematical operations like dot product, matrix addition, scalar multiplication, and transpose.
- `neural_network.py`: Contains the implementation of a neural network with one hidden layer, including methods for forward propagation, computing cost, backward propagation, training the network, and making predictions.
- `main.py`: The main script that uses the functions defined in the other files to prepare data, train the neural network, and evaluate its performance.

## Getting Started

To get started with CatDogNet, you'll need to install Python and the necessary libraries. You can install them with pip:

```bash
pip install -r requirements.txt
```

Then, you can clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/CatDogNet.git
```

Navigate to the project directory and run the main script:

```bash
cd CatDogNet
python main.py
```

## Contributing

Contributions to CatDogNet are welcome! Please feel free to open an issue or submit a pull request.
