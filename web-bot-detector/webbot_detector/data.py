import torch
from torch.utils.data import Dataset

class WebTrafficDataset(Dataset):
    def __init__(self, tokenizer, max_length=128, data=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_columns = [
            'NUMBER_OF_REQUESTS', 'TOTAL_DURATION', 'AVERAGE_TIME',
            # ... list of all 30 features ...
        ]

    def __len__(self):
        return 1 if isinstance(self.data, dict) else len(self.data)

    def __getitem__(self, idx):
        # Format data into prompt for model
        data = self.data if isinstance(self.data, dict) else self.data.iloc[idx]
        prompt_lines = [
            "You are an advanced cybersecurity analyst specializing in web bot detection.",
            "Analyze the following web traffic data and determine whether it indicates human activity or a bot:"
        ]
        for feature in self.feature_columns:
            value = data.get(feature, 0) if isinstance(data, dict) else data[feature]
            prompt_lines.append(f"- {feature}: {value}")
        prompt_lines.append("Based on these attributes, classify the request as 'Bot' or 'Human'.")
        prompt = "\n".join(prompt_lines)
        
        # Convert to model inputs
        encoding = self.tokenizer(prompt, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding