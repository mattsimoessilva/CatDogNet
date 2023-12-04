import tensorflow as tf
from data_loader import image_generator, get_files
from model import create_model

# Get the file paths
train_files = get_files('train/train')
test_files = get_files('test/test')

# Create the infinite image generators for training
train_generator = image_generator(train_files, batch_size=32)
test_generator = image_generator(test_files, batch_size=32)

# Create the model
model = create_model()

# Compile the model with a suitable optimizer and learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
steps_per_epoch = int(len(train_files) / 32)
validation_steps = int(len(test_files) / 32)

# Train the model using the generator
epochs = 0
while True:  # Set your desired number of epochs
    model.fit(train_generator,
              steps_per_epoch=steps_per_epoch,
              epochs=1,  # Train for 1 epoch at a time
              validation_data=test_generator,
              validation_steps=validation_steps)

    # Check the accuracy and stop if it's above 90%
    accuracy = model.evaluate(test_generator, steps=validation_steps)[1]
    print(f"Validation accuracy after {epochs + 1} epoch(s): {accuracy}")

    if accuracy > 0.9:
        print("\nReached 90% accuracy, stopping training.")
        break

    epochs += 1

# Save the model
model.save('alt_model.h5')
