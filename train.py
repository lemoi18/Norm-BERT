import yaml
import os
import os.path as osp
import shutil
import pickle
from datasets import load_from_disk
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AlbertConfig, AlbertModel, AdamW
from tqdm import tqdm
from dataloader import build_dataloader
from utils import length_to_mask
from model import MultiTaskModel
from torch import nn
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb


# Load configuration
config_path = "Configs/config.yml"
config = yaml.safe_load(open(config_path))

# Load token maps
with open(config['dataset_params']['token_maps'], 'rb') as handle:
    token_maps = pickle.load(handle)

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained('NbAiLab/nb-bert-large')  # Norwegian tokenizer

import yaml
import os
import os.path as osp
import shutil
import pickle
from datasets import load_from_disk
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AlbertConfig, AlbertModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from dataloader import build_dataloader
from utils import length_to_mask
from torch import nn
from accelerate import Accelerator
import wandb

# Load configuration
config_path = "Configs/config.yml"
config = yaml.safe_load(open(config_path))

# Load token maps
with open(config['dataset_params']['token_maps'], 'rb') as handle:
    token_maps = pickle.load(handle)

# Tokenizer initialization

from transformers import AutoTokenizer, AlbertConfig, AlbertModel, AutoModel, AdamW
import torch
from torch import nn
import torch.nn.functional as F
from accelerate import Accelerator
import os
import wandb
import shutil


# Define the student model (ALBERT) with multi-task prediction
class MultiTaskModel(nn.Module):
    def __init__(self, model, num_tokens=178, num_vocab=84827, hidden_size=768):
        super().__init__()
        self.encoder = model
        self.mask_predictor = nn.Linear(hidden_size, num_tokens)  # Token-level prediction
        self.word_predictor = nn.Linear(hidden_size, num_vocab)   # Vocabulary-level prediction
        
        # Apply Xavier initialization to the linear layers
        nn.init.xavier_uniform_(self.mask_predictor.weight)
        nn.init.xavier_uniform_(self.word_predictor.weight)
        
        # Optional: Initialize biases to zero for stability
        if self.mask_predictor.bias is not None:
            nn.init.zeros_(self.mask_predictor.bias)
        if self.word_predictor.bias is not None:
            nn.init.zeros_(self.word_predictor.bias)

    def forward(self, words, attention_mask=None):
        # Ensure the shapes match expectations before proceeding
        assert words.dim() == 2, f"Expected words dim=2, got {words.dim()} with shape {words.shape}"
        if attention_mask is not None:
            assert attention_mask.shape == words.shape, f"Shape mismatch: attention_mask {attention_mask.shape} vs words {words.shape}"

        output = self.encoder(words, attention_mask=attention_mask)
        
        tokens_pred = self.mask_predictor(output.last_hidden_state)
        words_pred = self.word_predictor(output.last_hidden_state)

        # Confirm output shapes
        assert tokens_pred.shape[:2] == words.shape, f"tokens_pred shape {tokens_pred.shape} does not match words shape {words.shape}"
        assert words_pred.shape[:2] == words.shape, f"words_pred shape {words_pred.shape} does not match words shape {words.shape}"

        return tokens_pred, words_pred


# Use train2() with debugging code as outlined in the previous response


# Define distillation loss function
# Define distillation loss function with additional stabilization techniques
def distillation_loss(student_logits, teacher_logits, target_labels, temperature=1.0, alpha=0.5, epsilon=1e-10):
    # Reshape logits for compatibility
    student_logits = student_logits.view(-1, student_logits.size(-1))
    teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))

    # Verify logits are finite with detailed debugging
    assert torch.isfinite(student_logits).all(), \
        f"NaN or Inf in student_logits:\nmin={student_logits.min()}, max={student_logits.max()}, mean={student_logits.mean()}"
    assert torch.isfinite(teacher_logits).all(), \
        f"NaN or Inf in teacher_logits before processing:\nmin={teacher_logits.min()}, max={teacher_logits.max()}, mean={teacher_logits.mean()}"

    # Cross-entropy loss for ground truth labels
    ce_loss = nn.CrossEntropyLoss()(student_logits, target_labels.view(-1))
    assert torch.isfinite(ce_loss), f"NaN or Inf in ce_loss:\nce_loss={ce_loss}"

    # Apply temperature scaling and clamping for teacher logits
    teacher_logits_temp = torch.clamp(teacher_logits / temperature, min=-50, max=50)
    student_logits_temp = torch.clamp(student_logits / temperature, min=-50, max=50)

    # Calculate log probabilities
    student_log_probs = F.log_softmax(student_logits_temp, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits_temp, dim=-1)

    # Convert teacher log probs back to probabilities and apply stabilization
    teacher_probs = torch.exp(teacher_log_probs)
    teacher_probs = torch.nan_to_num(teacher_probs, nan=epsilon, posinf=epsilon, neginf=epsilon).clamp(min=epsilon)

    # Final assertions for stability
    assert torch.isfinite(student_log_probs).all(), \
        f"NaN or Inf in student_log_probs:\nmin={student_log_probs.min()}, max={student_log_probs.max()}, mean={student_log_probs.mean()}"
    assert torch.isfinite(teacher_probs).all(), \
        f"NaN or Inf in teacher_probs after stabilization:\nmin={teacher_probs.min()}, max={teacher_probs.max()}, mean={teacher_probs.mean()}"

    # Compute KL divergence
    kl_loss_elementwise = teacher_probs * (teacher_log_probs - student_log_probs)
    kl_loss = kl_loss_elementwise.sum(dim=-1).mean()
    assert torch.isfinite(kl_loss), f"NaN or Inf in kl_loss:\nmin={kl_loss_elementwise.min()}, max={kl_loss_elementwise.max()}, mean={kl_loss_elementwise.mean()}"

    # Debugging outputs
    print(f"ce_loss: {ce_loss.item()}, kl_loss: {kl_loss.item()}")

    # Return combined loss
    return alpha * ce_loss + (1 - alpha) * kl_loss * (temperature ** 2)


from jiwer import wer  # Install jiwer if not already available: `pip install jiwer`

def calculate_wer(predictions, references):
    """
    Calculate the WER for a batch of predictions and references.
    Arguments:
        predictions (list of str): Decoded predictions from the model.
        references (list of str): Ground truth sentences.
    Returns:
        float: WER score for the batch.
    """
    return wer(references, predictions)








class TeacherModelProjection(nn.Module):
    def __init__(self, model_name='NbAiLab/nb-bert-large'):
        super(TeacherModelProjection, self).__init__()
        self.teacher_model = AutoModel.from_pretrained(model_name)
        # Retrieve hidden_size from the teacher model's config
        self.projection = nn.Linear(self.teacher_model.config.hidden_size, self.teacher_model.config.vocab_size)
    
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():  # Freeze teacher model for distillation
            teacher_output = self.teacher_model(input_ids, attention_mask=attention_mask)
        
        # Apply the projection to map to vocabulary size
        teacher_logits = self.projection(teacher_output.last_hidden_state)
        return teacher_logits



# Example configuration addition in config.yml
# use_distillation: true  # Set to false to disable distillation

def train2():
    criterion = nn.CrossEntropyLoss()  # Loss function for supervised task (vocab, token predictions)
    
    best_loss = float('inf')
    curr_steps = 0
    
    # Check the distillation flag in the configuration
    use_distillation = config.get("use_distillation", False)

    log_dir = config['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    accelerator = Accelerator(mixed_precision=config['mixed_precision'], split_batches=False)

    batch_size = config["batch_size"]

        # Get the number of epochs from config or use a default
    num_epochs = config.get('num_epochs', 10)

    train_loader = build_dataloader("processed_text_data.pkl", 
                                    batch_size=batch_size, 
                                    num_workers=0, 
                                    dataset_config=config['dataset_params'])
    print(f"Dataset length: {len(train_loader.dataset)}")

    albert_base_configuration = AlbertConfig(**config['model_params'])
    print("Vocabulary size:", albert_base_configuration.vocab_size)

    student_model = AlbertModel(albert_base_configuration)
    student_model = MultiTaskModel(student_model, 
                                    num_vocab=50000,  # Aligns with tokenizer's vocabulary size
                                    num_tokens=50000,
                                   hidden_size=config['model_params']['hidden_size'])

    # Load teacher model only if distillation is enabled
    if use_distillation:
        teacher_model = TeacherModelProjection('NbAiLab/nb-bert-large')
    
    optimizer = AdamW(student_model.parameters(), lr=1e-6)

    # Prepare models, optimizer, and dataloader for training
    if use_distillation:
        student_model, teacher_model, optimizer, train_loader = accelerator.prepare(
            student_model, teacher_model, optimizer, train_loader
        )
    else:
        student_model, optimizer, train_loader = accelerator.prepare(
            student_model, optimizer, train_loader
        )

    # Initialize WandB for logging
    wandb.init(project='NORM-BERT', config=config)
    wandb.watch(student_model, log='all')
    wandb.config.update(config)
    accelerator.print('Start training...')
    epoch_wer = []  # Initialize once at the start of the function

    running_loss = 0
    print(len(train_loader.dataset))
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        batch_wer = []  # WER for each batch within the epoch

        for _, batch in enumerate(train_loader):
            curr_steps += 1

            # Unpack batch data
            masked_tokens, words, labels, input_lengths, masked_indices = batch
            text_mask = length_to_mask(torch.tensor(input_lengths)).to(accelerator.device)
            masked_indices_tensor = masked_indices.to(accelerator.device)
            
            # Forward pass
            tokens_pred, words_pred = student_model(masked_tokens, attention_mask=text_mask)
            
            # Get masked tokens predictions and their corresponding labels
            tokens_pred_flat = tokens_pred.view(-1, tokens_pred.size(-1))[masked_indices_tensor.flatten()]
            labels_flat = labels.view(-1)[masked_indices_tensor.flatten()]
            
            # Compute the token-level loss only on masked tokens
            loss_token = criterion(tokens_pred_flat, labels_flat)
            
            # Similarly, filter the vocabulary-level predictions for the masked tokens
            words_pred_flat = words_pred.view(-1, words_pred.size(-1))[masked_indices_tensor.flatten()]
            words_labels_flat = words.view(-1)[masked_indices_tensor.flatten()]
            
            # Compute vocabulary-level loss only on masked tokens
            loss_vocab = criterion(words_pred_flat, words_labels_flat)


            # Apply distillation if enabled
            if use_distillation:
                with torch.no_grad():
                    teacher_logits = teacher_model(masked_tokens, attention_mask=text_mask)
                distill_loss_vocab = distillation_loss(words_pred, teacher_logits, words)
                distill_loss_token = distillation_loss(tokens_pred, teacher_logits, labels)
                loss = loss_vocab + loss_token + distill_loss_vocab + distill_loss_token
            else:
                loss = loss_vocab + loss_token

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            running_loss += loss.item()

            # Calculate WER for the batch
            decoded_preds = tokenizer.batch_decode(words_pred.argmax(dim=-1).cpu().numpy(), skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(words.cpu().numpy(), skip_special_tokens=True)
            batch_wer.append(calculate_wer(decoded_preds, decoded_labels))

            # Logging and WER Calculation
            if (curr_steps + 1) % config['log_interval'] == 0:
                print(f"Sample Prediction: {decoded_preds[0]}")
                print(f"Sample Reference: {decoded_labels[0]}")
                avg_batch_wer = sum(batch_wer) / len(batch_wer) if batch_wer else 0
                accelerator.print(f'Step {curr_steps + 1}, Epoch {epoch + 1}, Loss: {loss.item()}, Batch WER: {avg_batch_wer},vocab_loss: {loss_vocab.item()},token_loss: {loss_token.item()}')
                wandb.log({
                    'train_step': curr_steps + 1,
                    'loss': loss.item(),
                    'vocab_loss': loss_vocab.item(),
                    'token_loss': loss_token.item(),
                    'batch_wer': avg_batch_wer
                })
                batch_wer.clear()  # Reset batch_wer after each log interval

                if use_distillation:
                    wandb.log({
                        'distill_loss_vocab': distill_loss_vocab.item(),
                        'distill_loss_token': distill_loss_token.item()
                    })

            # Checkpoint saving
            if (curr_steps + 1) % config['save_interval'] == 0:
                accelerator.print(f'Saving checkpoint at step {curr_steps + 1}')
                save_dict = {
                    'student_model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                if use_distillation:
                    save_dict['teacher_model_state_dict'] = teacher_model.state_dict()
                accelerator.save(save_dict, os.path.join(log_dir, f'step_{curr_steps + 1}.pt'))

            # Exit if maximum training steps reached
            if curr_steps >= config['num_steps']:
                print("Reached maximum training steps.")
                return  # Exit the function

        # Calculate and log epoch WER at end of each epoch
        epoch_avg_wer = sum(batch_wer) / len(batch_wer) if batch_wer else 0
        epoch_wer.append(epoch_avg_wer)
        wandb.log({'epoch': epoch + 1, 'epoch_wer': epoch_avg_wer})
        print(f"Completed epoch {epoch + 1}/{num_epochs} with Avg WER: {epoch_avg_wer}")

if __name__ == '__main__':
    train2()
