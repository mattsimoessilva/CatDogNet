import os
import cv2
import numpy as np

def image_generator(files, batch_size=32):
    while True:
        # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a=files, size=batch_size)
        batch_input = []
        batch_output = []

        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            input = cv2.imread(input_path)
            output = [1, 0] if 'cat' in input_path else [0, 1]

            input = cv2.resize(input, (64, 64))
            input = input.flatten()


            batch_input.append(input)
            batch_output.append(output)
        # Return a tuple of (input, output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield (batch_x, batch_y)

def finite_image_generator(files, batch_size=32):
    num_files = len(files)
    num_batches = num_files // batch_size

    for i in range(num_batches):
        # Select files (paths/indices) for the batch
        batch_paths = files[i*batch_size : (i+1)*batch_size]
        batch_input = []
        batch_output = []

        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            input = cv2.imread(input_path)
            output = [1, 0] if 'cat' in input_path else [0, 1]

            input = cv2.resize(input, (64, 64))
            input = input.flatten()

            batch_input.append(input)
            batch_output.append(output)

        # Return a tuple of (input, output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield (batch_x, batch_y)

