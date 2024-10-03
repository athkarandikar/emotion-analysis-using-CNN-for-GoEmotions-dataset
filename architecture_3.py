import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization

emotions = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Load train, test, and validation (dev) data
train_df = pd.read_csv('train.tsv', sep='\t', header=None, names=['text', 'emotion_ids', 'comment_id'])
test_df = pd.read_csv('test.tsv', sep='\t', header=None, names=['text', 'emotion_ids', 'comment_id'])
val_df = pd.read_csv('dev.tsv', sep='\t', header=None, names=['text', 'emotion_ids', 'comment_id'])

# Function to convert emotion_ids into a binary label vector
def process_emotion_ids(emotion_ids_str, num_emotions):
    emotion_ids = list(map(int, emotion_ids_str.split(',')))  # Split and convert to integers
    label_vector = [0] * num_emotions
    for idx in emotion_ids:
        label_vector[idx] = 1  # Set corresponding emotion index to 1
    return label_vector

# Apply the function to convert emotion_ids to binary vectors for each DataFrame
train_df['labels'] = train_df['emotion_ids'].apply(lambda x: process_emotion_ids(x, len(emotions)))
test_df['labels'] = test_df['emotion_ids'].apply(lambda x: process_emotion_ids(x, len(emotions)))
val_df['labels'] = val_df['emotion_ids'].apply(lambda x: process_emotion_ids(x, len(emotions)))

# Keep only text and labels
train_df = train_df[['text', 'labels']]
test_df = test_df[['text', 'labels']]
val_df = val_df[['text', 'labels']]

# Text preprocessing
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100

# Tokenizing text
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(train_df['text'].values)

train_sequences = tokenizer.texts_to_sequences(train_df['text'].values)
train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)

test_sequences = tokenizer.texts_to_sequences(test_df['text'].values)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

val_sequences = tokenizer.texts_to_sequences(val_df['text'].values)
val_data = pad_sequences(val_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Convert labels to numpy arrays
train_labels = np.array(train_df['labels'].tolist())
test_labels = np.array(test_df['labels'].tolist())
val_labels = np.array(val_df['labels'].tolist())

# Model architecture
model = Sequential()
model.add(Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM))  # Embedding layer
model.add(Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.001)))  # Convolutional layer with L2 regularization
model.add(BatchNormalization())  # Batch normalization
model.add(GlobalMaxPooling1D())  # Global max pooling
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))  # Fully connected layer with L2 regularization
model.add(Dense(len(emotions), activation='sigmoid'))  # Output layer for 28 emotions (sigmoid for multi-label classification)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(train_data, train_labels, epochs=7, batch_size=32, validation_data=(val_data, val_labels), callbacks=[early_stopping])

# Predict on test data
test_predictions = model.predict(test_data)
test_predictions = np.where(test_predictions > 0.5, 1, 0)  # Convert probabilities to binary labels

# Ensure labels and predictions are numpy arrays
test_labels = np.array(test_labels)
test_predictions = np.array(test_predictions)

# Evaluate using accuracy, precision, recall, and F1 score
accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(test_labels, test_predictions, average='micro')
recall = recall_score(test_labels, test_predictions, average='micro')
f1 = f1_score(test_labels, test_predictions, average='micro')

print(f"Test Accuracy: {accuracy}")
print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test F1 Score: {f1}")

# Function to predict emotions from a given text
def predict_emotions(text, tokenizer, model, max_sequence_length=100):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model.predict(padded_sequence)
    predicted_labels = np.where(prediction > 0.5, 1, 0).flatten()
    predicted_emotions = [emotions[i] for i in range(len(predicted_labels)) if predicted_labels[i] == 1]
    return predicted_emotions

text_input = "I am so excited for the concert tonight!"
predicted_emotions = predict_emotions(text_input, tokenizer, model)
print(f"Predicted Emotions: {predicted_emotions}")