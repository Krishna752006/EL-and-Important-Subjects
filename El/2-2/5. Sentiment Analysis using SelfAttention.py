import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

data = pd.read_csv('5. Tweets.csv', sep=',')

data = data.dropna(subset=["text", "airline_sentiment"])
data.shape

def simple_tokenize(text):
    return text.lower().split()

word_to_index = {}
tokenized_texts = []

for text in data["text"]:
    tokens = simple_tokenize(text)
    tokenized_texts.append(tokens)

    for token in tokens:
        if token not in word_to_index:
            word_to_index[token] = len(word_to_index) + 1  # Index starts at 1

sequences = []
for tokens in tokenized_texts:
    sequence = [word_to_index.get(token, 0) for token in tokens]
    sequences.append(sequence)

padded_sequences = np.array([seq + [0] * (max_len - len(seq)) for seq in sequences])

labels = data["airline_sentiment"].map({"positive": 2, "neutral": 1, "negative": 0}).values

X = torch.tensor(padded_sequences, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

class ScaledDotAttention(nn.Module): 
    def __init__(self, hidden_size):
        super(ScaledDotAttention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, Q, K, V):

        attention_scores = torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(self.hidden_size)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        self.last_Q = Q
        self.last_K = K
        self.last_V = V

        return output

def test_scaled_dot_attention():

    hidden_size = 128
    attention_layer = ScaledDotAttention(hidden_size)
    
    # Create dummy Q, K, and V tensors for the test
    batch_size = 2
    seq_len = 4
    feature_size = hidden_size
    Q = torch.randn(batch_size, seq_len, feature_size)  # Query tensor
    K = torch.randn(batch_size, seq_len, feature_size)  # Key tensor
    V = torch.randn(batch_size, seq_len, feature_size)  # Value tensor

    # Perform the forward pass
    output = attention_layer(Q, K, V)

    # Assert that the output shape is correct: (batch_size, seq_len, feature_size)
    assert output.shape == (batch_size, seq_len, feature_size), f"Expected output shape (batch_size, seq_len, feature_size), but got {output.shape}"

# Run the test case
test_scaled_dot_attention()

class SentimentAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(SentimentAnalysisModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.attention = ScaledDotAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, 3)  # 3 classes: positive, neutral, negative

    def forward(self, x):
        embedded = self.embedding(x) 
        print("embedded",embedded.shape)
        #(batch_size, sequence_length, embedding_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # shape(batch_size, sequence_length, hidden_size) ; Hidden state ,# cell state
        print("lstm_out",lstm_out.shape)
        # Apply scaled dot-product attention
        attention_out = self.attention(lstm_out, lstm_out, lstm_out)  # Q, K, V are lstm_out
        print("attention_out",attention_out.shape)
        out = attention_out[:, -1, :]  # Get the output of the last time step
        out = self.fc(out)
        return out

def test_sentiment_analysis_model():
    # Initialize the model with vocab_size, embedding_dim, and hidden_size
    vocab_size = 5000  # Example vocab size
    embedding_dim = 100
    hidden_size = 128
    model = SentimentAnalysisModel(vocab_size, embedding_dim, hidden_size)

    # Create dummy input data (e.g., batch size = 2, sequence length = 5)
    batch_size = 2
    seq_len = 5
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))  # Random token indices

    # Perform a forward pass
    output = model(dummy_input)

    embedded = model.embedding(dummy_input)
    lstm_out, (h_n, c_n) = model.lstm(embedded)
    attention_out = model.attention(lstm_out, lstm_out, lstm_out)
    
    assert attention_out.shape == torch.Size([batch_size, seq_len, hidden_size]), \
        f"Expected attention_out shape (batch_size, seq_len, hidden_size), but got {attention_out.shape}"

    print(f"Final output: {output.shape}")

# Run the test case
test_sentiment_analysis_model()

# For reference
vocab_size = len(word_to_index) + 1  # Include padding token
embedding_dim = 100
hidden_size = 128

print("Total unique word indexs: ", len(word_to_index))
print("Total unique words: ",vocab_size)

model = SentimentAnalysisModel(vocab_size, embedding_dim, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0, 0

        print(f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (inputs, labels) in enumerate(train_loader):

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += (outputs.argmax(1) == labels).sum().item()

            if batch_idx == 0:
                attention_layer = model.attention
                Q, K, V = attention_layer.last_Q, attention_layer.last_K, attention_layer.last_V
                print(f"Q shape: {Q.shape}")
                print(f"K shape: {K.shape}")
                print(f"V shape: {V.shape}")

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader.dataset)

        print(f"Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                val_acc += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

def create_dummy_dataset(batch_size=64, seq_len=5, vocab_size=5000, num_samples=1000):
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, 3, (num_samples,))
    dataset = TensorDataset(X, y)
    return dataset

def test_train_model():
    vocab_size = 5000
    embedding_dim = 100
    hidden_size = 128
    model = SentimentAnalysisModel(vocab_size, embedding_dim, hidden_size)

    # Create dummy data
    train_dataset = create_dummy_dataset()
    val_dataset = create_dummy_dataset()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 2
    train_model(model, train_loader, val_loader, epochs)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        assert outputs.shape == (inputs.shape[0], 3), f"Expected output shape (batch_size, 3), but got {outputs.shape}"

        loss = criterion(outputs, labels)
        loss.backward()

        if batch_idx >= 2:  # Run a few batches and then break
            break

    print(" Test Case 4 Passed:")

test_train_model()

epochs = 5
train_model(model, train_loader, test_loader, epochs)

model.eval()
correct, total = 0, 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

def predict_sentiment(model, text, word_to_index):
    text_indices = [[word_to_index.get(word, 0) for word in simple_tokenize(t)] for t in text]

    max_len = max(map(len, text_indices))
    text_indices = [seq + [0] * (max_len - len(seq)) for seq in text_indices]

    text_tensor = torch.tensor(text_indices, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        output = model(text_tensor)

    sentiment_labels = ["Neutral Comment", "Negative Comment", "Positive Comment"]
    return f"Predicted Sentiment: {sentiment_labels[output.argmax(dim=1).item()]}"

t1 = ['Nothing special, but no complaints.']
t2 = ['I am too excited ']  # Very happy to hear you
t3 = ['Something went wrong']  # It's disappointing

# Choose text for prediction
unseen_text = t3 

result = predict_sentiment(model, unseen_text , word_to_index)
print(result)