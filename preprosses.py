from transformers import AutoTokenizer
import pickle

# Initialize the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('NbAiLab/nb-bert-large')
def read_data(file_path):
    """
    Reads the text file and extracts the relevant fields: file name, unnormalized text, and speaker ID.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) == 3:
            file_name = parts[0].strip()  # file_name can be used later if needed for audio reference
            text = parts[1].strip()  # Extract the text (unnormalized or normalized)
            speaker_id = int(parts[2].strip())  # Speaker ID or label (if needed)
            data.append({'text': text,
                         'file_name':file_name
                        })
    
    return data

def merge_unnormalized_normalized(unnormalized_data, normalized_data):
    """
    Merges the unnormalized and normalized datasets into a single dataset with matched text and token-level alignment.
    """
    merged_data = []
    
    for unnormalized, normalized in zip(unnormalized_data, normalized_data):
        if unnormalized['file_name'] == normalized['file_name']:
            # Tokenize both texts
            unnorm_tokens = tokenizer(unnormalized['text'], add_special_tokens=True, return_tensors='pt')['input_ids'][0]
            norm_tokens = tokenizer(normalized['text'], add_special_tokens=True, return_tensors='pt')['input_ids'][0]
            
            # Alignment tracking
            alignment = []
            norm_idx = 0
            
            for unnorm_idx, unnorm_token in enumerate(unnorm_tokens):
                corresponding_norm_indices = []
                
                # Find matching normalized tokens
                while norm_idx < len(norm_tokens):
                    if norm_tokens[norm_idx] == unnorm_token:
                        corresponding_norm_indices.append(norm_idx)
                        norm_idx += 1
                        break  # Move to next unnorm token
                    else:
                        corresponding_norm_indices.append(norm_idx)
                    norm_idx += 1
                
                # Add the unnorm token with its corresponding indices in normalized text
                if corresponding_norm_indices:
                    alignment.append((unnorm_idx, corresponding_norm_indices))
            
            # Add tokenized data and alignment to merged data
            merged_data.append({
                'file_name': unnormalized['file_name'],
                'unnormalized_text': unnormalized['text'],
                'normalized_text': normalized['text'],
                'unnormalized_tokens': unnorm_tokens.tolist(),
                'normalized_tokens': norm_tokens.tolist(),
                'alignment': alignment
            })
    
    return merged_data

# Example call to merge and save
unnormalized_file = 'outputbooks.txt'
normalized_file = 'normalized_output.txt'
unnormalized_data = read_data(unnormalized_file)
normalized_data = read_data(normalized_file)

# Merging with alignment
merged_data_with_alignment = merge_unnormalized_normalized(unnormalized_data, normalized_data)

# Saving to pickle
output_file = "processed_text_with_alignment.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(merged_data_with_alignment, f)
