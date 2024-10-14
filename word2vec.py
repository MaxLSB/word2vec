import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
from transformers import BertTokenizer

############################################################################################################

# Dataset and Model classes


class DocumentDataset(Dataset):
    def __init__(self, words, contexts):
        self.words = words
        self.contexts = contexts

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return (torch.tensor(self.contexts[idx]), torch.tensor(self.words[idx]))


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        # We create two embedding layers and we set the padding_idx to 0
        self.word_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.context_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    def forward(self, word_ids, context_ids):
        w = self.word_embed(word_ids)
        C = self.context_embed(context_ids)
        dot_product = torch.bmm(w.unsqueeze(1), C.transpose(1, 2)).squeeze(1)
        return dot_product


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


def extract_words_contexts(text_ids, R):
    words = []
    contexts = []
    for i in range(len(text_ids)):
        words.append(text_ids[i])
        context = []
        for j in range(max(0, i - R), min(len(text_ids), i + R + 1)):
            if i != j:
                context.append(text_ids[j])
        if (
            len(context) < 2 * R
        ):  # Adding a padding token of id 0 when the context is less than 2R
            context += [0] * (2 * R - len(context))
        contexts.append(context)
    return words, contexts


def flatten_dataset_to_lists(dataset, R):
    words = []
    contexts = []
    for example in dataset:
        w, c = extract_words_contexts(example["review_ids"], R)
        words.extend(w)
        contexts.extend(c)
    return contexts, words


def collate_fn(batch, K, vocab):
    positive_context_ids, word_ids = zip(*batch)

    # Convert to torch tensors
    word_ids = torch.tensor(word_ids)
    positive_context_ids = torch.stack(
        [torch.tensor(ids) for ids in positive_context_ids]
    )

    # Generate negative samples
    batch_size, context_size = positive_context_ids.shape
    vocab_size = len(vocab)
    negative_context_ids = torch.randint(
        0, vocab_size, (batch_size, K * context_size), dtype=torch.long
    )

    return {
        "word_ids": word_ids,
        "positive_context_ids": positive_context_ids,
        "negative_context_ids": negative_context_ids,
    }


def save_model(model, model_path):
    torch.save(
        {
            "word_embed": model.word_embed.weight.data.clone(),  # Save word embeddings
            "context_embed": model.context_embed.weight.data.clone(),  # Save context embeddings
        },
        model_path,
    )


############################################################################################################

# Main function


def main():

    # Hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate = 0.001
    embedding_dim = 100
    B = 256  # Batch size
    E = 5  # Number of epochs
    K = 2  # Factor for negative samples
    R = 10  # Context window size

    # Loading the dataset & Tokenizer
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

    # Model
    model = Word2Vec(vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loading the data
    print("Loading the data...")
    train_contexts, train_words = flatten_dataset_to_lists(document_train_set, R)
    val_contexts, val_words = flatten_dataset_to_lists(document_valid_set, R)
    train_set = DocumentDataset(train_words, train_contexts)
    val_set = DocumentDataset(val_words, val_contexts)
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=B,
        collate_fn=lambda batch: collate_fn(batch, K=K, vocab=tokenizer.vocab),
    )
    val_data_loader = DataLoader(
        dataset=val_set,
        batch_size=B,
        collate_fn=lambda batch: collate_fn(batch, K=K, vocab=tokenizer.vocab),
    )

    print(f"Number of Parameters: {sum(p.numel() for p in model.parameters())}")

    # Training the model
    model.train()
    for epoch in range(E):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}/{E}"):
            word_ids = batch["word_ids"].to(device)
            positive_context_ids = batch["positive_context_ids"].to(device)
            negative_context_ids = batch["negative_context_ids"].to(device)

            # Outputs of the model
            logits_pos = model(word_ids, positive_context_ids)
            logits_neg = model(word_ids, negative_context_ids)

            # Loss
            loss_pos = F.logsigmoid(logits_pos).mean()
            loss_neg = F.logsigmoid(-logits_neg).mean()
            loss = -loss_pos - loss_neg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Testing the model on the validation set
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for batch in val_data_loader:
                word_ids = batch["word_ids"].to(device)
                positive_context_ids = batch["positive_context_ids"].to(device)
                negative_context_ids = batch["negative_context_ids"].to(device)

                # Outputs of the model
                logits_pos = model(word_ids, positive_context_ids)
                logits_neg = model(word_ids, negative_context_ids)

                # Accuracy, we don't count the padding tokens
                correct_predictions += (
                    ((F.sigmoid(logits_pos) > 0.5) & (positive_context_ids != 0))
                    .sum()
                    .item()
                )
                correct_predictions += (
                    ((F.sigmoid(logits_neg) < 0.5) & (negative_context_ids != 0))
                    .sum()
                    .item()
                )
                total_pos = (positive_context_ids != 0).sum().item()
                total_neg = (negative_context_ids != 0).sum().item()
                total_predictions += total_pos + total_neg

            accuracy = correct_predictions / total_predictions

        print(
            f"Epoch {epoch + 1} | Training Loss: {total_loss/len(train_dataloader)}, Test Accuracy: {accuracy:.4f}"
        )

    print("Training completed!")

    # Save the model
    file_name = (
        f"model_dim-{embedding_dim}_radius-{R}_ratio-{K}-batch-{B}-epoch-{E}.ckpt"
    )
    save_model(model, file_name)
    print(f"Model saved as {file_name}")


if __name__ == "__main__":
    main()
