import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from .data import WebTrafficDataset

class BotDetectionModel(nn.Module):
    def __init__(self, model_name, num_labels=2, mlp_hidden_size=256, dropout_prob=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

class BotDetector:
    def __init__(self, model_path, model_name="bert-base-uncased", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BotDetectionModel(model_name=model_name).to(self.device)
        
        # Load the trained model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'],strict=False)
        self.model.eval()
        
        print(f"Bot detector initialized using {model_name} on {self.device}")

    def predict(self, traffic_data):
        """
        Predict if traffic is from a bot or human
        
        Args:
            traffic_data: Dictionary or DataFrame containing traffic features
            
        Returns:
            Dictionary with prediction result (is_bot, confidence)
        """
        # Create dataset from input data
        dataset = WebTrafficDataset(tokenizer=self.tokenizer, data=traffic_data)
        
        # Get encoding
        encoding = dataset[0]
        input_ids = encoding['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = encoding['attention_mask'].unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1)
            is_bot = torch.argmax(probs, dim=1).item() == 1
            confidence = probs[0][1].item() if is_bot else probs[0][0].item()
        
        return {
            "is_bot": bool(is_bot),
            "confidence": float(confidence),
            "label": "Bot" if is_bot else "Human"
        }