import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tabulate import tabulate
from datasets import load_dataset
from tqdm import tqdm
from transformers import BertTokenizer


############################################################################################################

# Dataset and Model classes


class ClassificationDataset(Dataset):
    def __init__(self, dataset, max_length):
        self.dataset = dataset
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if len(sample["review_ids"]) < self.max_length:
            pad = self.max_length - len(sample["review_ids"])
            sample["review_ids"].extend([0] * pad)

        return (torch.tensor(sample["review_ids"]), torch.tensor(sample["label"]))


class ConvolutionModel(nn.Module):
    def __init__(self, embedding_dim, conv_dim, vocab_size, num_classes):
        super(ConvolutionModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, conv_dim, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(conv_dim, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)  # (batch_size, embed_size, sequence_length)
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.fc(x)
        return x


############################################################################################################

# Utility functions


def preprocessing_fn(x, tokenizer):
    x["review_ids"] = tokenizer(
        x["review"],
        add_special_tokens=False,
        truncation=True,
        max_length=256,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    x["label"] = 0 if x["sentiment"] == "negative" else 1
    return x


def load_model(model_path, model):
    # Load the saved Word2Vec embeddings
    checkpoint = torch.load(model_path)
    embeddings = checkpoint["context_embed"]

    # Load the embeddings from the word2vec pre-trained model
    model.embed.weight.data.copy_(embeddings)

    return model


############################################################################################################

# Main function


def main():

    # Hyperparameters
    embedding_dim = 100  # Same as the word2vec model
    conv_dim = 100
    num_classes = 2
    num_epochs = 5
    lr = 0.001
    model_path = "model_dim-100_radius-10_ratio-2-batch-256-epoch-5.ckpt"
    load_embeddings = True  # Load the embeddings from the word2vec model if True

    # Load the dataset & tokenizer
    dataset = load_dataset("scikit-learn/imdb", split="train")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    n_samples = 5000  # the number of training example
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(n_samples))  # Select 5000 samples
    dataset = dataset.map(lambda x: preprocessing_fn(x, tokenizer))
    dataset = dataset.remove_columns(["review", "sentiment"])  # Remove useless columns

    # Split the train and validation set
    split = dataset.train_test_split(test_size=0.1, seed=42)
    document_train_set = split["train"]
    document_valid_set = split["test"]

    # Loading the dataset
    train_set = ClassificationDataset(document_train_set, max_length=256)
    val_set = ClassificationDataset(document_valid_set, max_length=256)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    # Training the model with loaded embeddings
    model = ConvolutionModel(
        embedding_dim=embedding_dim,
        conv_dim=conv_dim,
        vocab_size=tokenizer.vocab_size,
        num_classes=num_classes,
    )
    # Load the embeddings from the word2vec if load_embeddings is True
    if load_embeddings:
        model = load_model(model_path, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{num_epochs}"):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for x, y in val_loader:
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                val_acc += (y_pred.argmax(1) == y).sum().item()

            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_acc / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    print("Training completed!")


if __name__ == "__main__":
    main()
