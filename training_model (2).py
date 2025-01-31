import torch
from torch.utils.data import DataLoader, Dataset
from mistral_inference.model import Transformer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from torch.nn.utils.rnn import pad_sequence
import logging
import os
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Normalization Layer
class CustomNormalizationLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, variance_epsilon: float = 1e-5):
        super(CustomNormalizationLayer, self).__init__()
        self.variance_epsilon = variance_epsilon
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# Define the ILLMModel class
class ILLMModel:
    def __init__(self, model_path: str):
        self.tokenizer = MistralTokenizer.from_file(os.path.join(model_path, "tokenizer.model.v3"))
        try:
            self.model = Transformer.from_folder(model_path).to(device=device, dtype=torch.float16)
            hidden_size = getattr(self.model, 'hidden_size', 1024)
            self.model.norm = CustomNormalizationLayer(hidden_size=hidden_size)
        except RuntimeError as e:
            logging.error("Switching to CPU due to memory issues: %s", e)
            self.model = Transformer.from_folder(model_path).to(device=device)

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def save_model(self, save_path: str) -> None:
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, "model_weights.pth"))
        try:
            with open(os.path.join(save_path, "tokenizer.model.v3"), "wb") as f:
                f.write(self.tokenizer.save())
        except AttributeError:
            logging.warning("Tokenizer does not have a save method, skipping tokenizer save.")

    def load_fine_tuned_model(self, fine_tuned_model_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(fine_tuned_model_path, "model_weights.pth")))
        self.model.to(device=device, dtype=torch.float16)

# Define the Dataset class
class ConversationDataset(Dataset):
    def __init__(self, conversations: List[str], summaries: List[str], tokenizer: MistralTokenizer):
        self.conversations = conversations
        self.summaries = summaries
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conversation = self.conversations[idx]
        summary = self.summaries[idx]

        conversation_message = ChatCompletionRequest(messages=[UserMessage(content=conversation)])
        summary_message = ChatCompletionRequest(messages=[UserMessage(content=summary)])

        input_ids = self.tokenizer.encode_chat_completion(conversation_message).tokens
        label_ids = self.tokenizer.encode_chat_completion(summary_message).tokens

        return {
            "input_ids": torch.tensor(input_ids).long(),
            "labels": torch.tensor(label_ids).long()
        }

# Define the collate function for padding
def collate_fn(batch: List[Dict[str, torch.Tensor]], tokenizer: MistralTokenizer) -> Dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels
    }

# Define the training function
def train_model(model: ILLMModel, train_dataloader: DataLoader, epochs: int = 3, lr: float = 5e-5) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=lr)
    pad_token_id = getattr(model.tokenizer, 'pad_token_id', 0)

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            seqlens = (input_ids != pad_token_id).sum(dim=1)

            try:
                outputs = model.model(input_ids=input_ids, seqlens=seqlens)
                logging.info("Outputs shape: %s", str(outputs))
                logging.info("Type of outputs: %s, Length of outputs: %d", type(outputs), len(outputs) if isinstance(outputs, tuple) else 1)

                if isinstance(outputs, tuple):
                    logits = outputs[0]  # Modify this line if there are multiple outputs
                else:
                    logits = outputs

                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            except Exception as e:
                logging.error("Error during model forward pass: %s", e)
                logging.error("Batch index: %d, Input IDs shape: %s, Labels shape: %s",
                              batch_idx, input_ids.shape, labels.shape)

        avg_loss = total_loss / len(train_dataloader)
        logging.info("Epoch %d/%d completed. Average Loss: %.4f", epoch + 1, epochs, avg_loss)
        
def optimize_model():
    def optimize_model(model: ILLMModel, train_dataloader: DataLoader, epochs: int = 3, lr: float = 5e-5) -> None:
        model.train()
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        pad_token_id = getattr(model.tokenizer, 'pad_token_id', 0)

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_dataloader):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                seqlens = (input_ids != pad_token_id).sum(dim=1)

                try:
                    outputs = model.model(input_ids=input_ids, seqlens=seqlens)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs

                    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                except Exception as e:
                    logging.error("Error during model forward pass: %s", e)
                    logging.error("Batch index: %d, Input IDs shape: %s, Labels shape: %s",
                                  batch_idx, input_ids.shape, labels.shape)

            scheduler.step()
            avg_loss = total_loss / len(train_dataloader)
            logging.info("Epoch %d/%d completed. Average Loss: %.4f", epoch + 1, epochs, avg_loss)
                
# Define the main function
def main():
    conversations = [
        "Doctor: How are you feeling after the surgery? Patient: I'm feeling much better, but still a bit sore.",
        "Nurse: Please take your medications as prescribed. Patient: Sure, I will."
    ]
    summaries = [
        "The patient is feeling better but is still experiencing some soreness.",
        "The patient agreed to take medications as prescribed."
    ]

    model_path = "C:/DATA/7B_instruct/"
    model = ILLMModel(model_path)

    dataset = ConversationDataset(conversations, summaries, model.tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                                  collate_fn=lambda batch: collate_fn(batch, model.tokenizer))

    train_model(model, train_dataloader, epochs=3)

    fine_tuned_model_path = "C:/DATA/fine_tuned_model/"
    model.save_model(fine_tuned_model_path)
    logging.info("Fine-tuned model saved.")

if __name__ == "__main__":
    main()
#--- This is the training module 