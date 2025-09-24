from config import AppConfig, MODEL_CONFIG, AppConfig
import torch 
import torch.nn as nn
from modules.logger import logging, get_log_file_name

class SpamTrainer:
    def __init__(self, model , train_loader, val_loader, optimizer, device, num_epochs,eval_freq, eval_iter):
        self.model = model
        self.train_loader = train_loader 
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.eval_freq = eval_freq
        self.eval_iter = eval_iter
        self.train_losses, self.val_losses, self.train_accs, self.val_accs = [], [], [], []
        self.examples_seen, self.global_step = 0, -1
        

    def train_val(self):
        # Main training loop
        for epoch in range(self.num_epochs):
            self.model.train()  # Set model to training mode

            for input_batch, target_batch in self.train_loader:
                self.optimizer.zero_grad() # Reset loss gradients from previous batch iteration
                loss = self.calc_loss_batch(input_batch, target_batch)
                loss.backward() # Calculate loss gradients
                self.optimizer.step() # Update model weights using loss gradients
                self.examples_seen += input_batch.shape[0] # New: track examples instead of tokens
                self.global_step += 1

                # Optional evaluation step
                if self.global_step % self.eval_freq == 0:
                    self.train_loss, self.val_loss = self.evaluate_model(self.model)
                    self.train_losses.append(self.train_loss)
                    self.val_losses.append(self.val_loss)
                    logging.info(f"Ep {epoch+1} (Step {self.global_step:06d}): "
                        f"Train loss {self.train_loss:.3f}, Val loss {self.val_loss:.3f}")

            # Calculate accuracy after each epoch
            self.train_accuracy = self.calc_accuracy_loader(model=self.model, data_loader=self.train_loader, num_batches=self.eval_iter)
            self.val_accuracy = self.calc_accuracy_loader(model = self.model, data_loader=self.val_loader, num_batches=self.eval_iter)
            print(f"Training accuracy: {self.train_accuracy*100:.2f}% | ", end="")
            print(f"Validation accuracy: {self.val_accuracy*100:.2f}%")
            self.train_accs.append(self.train_accuracy)
            self.val_accs.append(self.val_accuracy)

        return self.train_losses, self.val_losses, self.train_accs, self.val_accs, self.examples_seen        


    #Define the cross entropy calculations 
    def calc_loss_batch(self, input_batch, target_batch):
        input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)
        logits = self.model(input_batch)[:, -1, :]
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
        return loss


    def calc_loss_loader(self, model, data_loader, num_batches=None):
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(input_batch, target_batch)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches
    
    def evaluate_model(self, model):
        model.eval()
        with torch.no_grad():
            train_loss = self.calc_loss_loader(model, self.train_loader, num_batches=self.eval_iter)
            val_loss = self.calc_loss_loader(model, self.val_loader, num_batches=self.eval_iter)
        model.train()
        return train_loss, val_loss

    def calc_accuracy_loader(self, model, data_loader, num_batches=None):
        model.eval()
        correct_predictions, num_examples = 0, 0

        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)

                with torch.no_grad():
                    logits = model(input_batch)[:, -1, :]  # Logits of last output token
                predicted_labels = torch.argmax(logits, dim=-1)

                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break
        return correct_predictions / num_examples