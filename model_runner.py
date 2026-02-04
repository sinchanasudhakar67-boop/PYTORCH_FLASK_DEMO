import torch
import torch.nn as nn


class TextGenerator(nn.Module):
    """LSTM-based text generator model."""
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.5):
        super(TextGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Only apply LSTM dropout if num_layers > 1 (PyTorch requires this)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=lstm_dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = self.dropout(x)
        x, (h_n, c_n) = self.lstm(x, hidden)
        x = self.dropout(x)
        x = self.fc(x)
        return x, (h_n, c_n)
    
    def init_hidden(self, batch_size, device='cpu'):
        """Initialize hidden and cell states."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


def load_model(path, vocab_size=None, map_location='cpu', device=None):
    """Load a PyTorch model from `path`.

    This helper tries `torch.jit.load` first (scripted models), then
    `torch.load`. If the file contains a `state_dict` (a dict) it
    instantiates a TextGenerator model and loads the state_dict into it.
    
    Args:
        path: Path to the model file
        vocab_size: Vocabulary size for the model (auto-detected if None)
        map_location: Where to load the model (deprecated, use device instead)
        device: Device to load the model to (cpu/cuda)
    """
    # Convert device to map_location if provided
    if device is not None:
        map_location = device
    
    # try loading as a TorchScript module
    try:
        model = torch.jit.load(path, map_location=map_location)
        return model
    except Exception:
        pass

    # fallback to torch.load
    obj = torch.load(path, map_location=map_location)
    # If it's a state dict, instantiate the model and load it
    if isinstance(obj, dict):
        # Auto-detect model architecture from state dict if vocab_size not provided
        if vocab_size is None:
            # Get vocab size from embedding weight shape
            if 'embedding.weight' in obj:
                vocab_size = obj['embedding.weight'].shape[0]
            else:
                vocab_size = 10000  # fallback default
        
        # Auto-detect embedding_dim from embedding weight shape
        embedding_dim = obj['embedding.weight'].shape[1]
        
        # Auto-detect hidden_dim from lstm weight_ih_l0 shape (second dimension)
        hidden_dim = obj['lstm.weight_ih_l0'].shape[0] // 4
        
        # Auto-detect num_layers by counting lstm layer keys
        num_layers = 1
        layer_idx = 1
        while f'lstm.weight_ih_l{layer_idx}' in obj:
            num_layers += 1
            layer_idx += 1
        
        model = TextGenerator(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        model.load_state_dict(obj)
        if device is not None:
            model = model.to(device)
        return model
    return obj


def generate_text(model, word_to_idx, idx_to_word, prompt, num_words=100, device='cpu', sequence_length=64):
    """Generate text from the model using the given prompt.
    
    Args:
        model: The trained TextGenerator model
        word_to_idx: Dictionary mapping words to indices
        idx_to_word: Dictionary mapping indices to words
        prompt: Starting text (word or phrase)
        num_words: Number of words to generate
        device: Device to run inference on ('cpu' or 'cuda')
        sequence_length: Length of sequence to feed to model
    
    Returns:
        Generated text as a string
    """
    model = model.to(device)
    model.eval()
    
    words = prompt.split()
    
    with torch.no_grad():
        for _ in range(num_words):
            # Take the last sequence_length words (or fewer if less available)
            input_words = words[-sequence_length:]
            
            # Convert words to indices
            try:
                input_indices = [word_to_idx.get(word, 0) for word in input_words]
            except Exception:
                # If word_to_idx is not available, return what we have
                return " ".join(words)
            
            # Convert to tensor
            input_tensor = torch.LongTensor(input_indices).unsqueeze(0).to(device)
            
            # Initialize hidden state
            h, c = model.init_hidden(1, device=device)
            
            # Get prediction
            output, (h, c) = model(input_tensor, (h, c))
            
            # Get the last output and find the word with highest probability
            next_token = output[0, -1, :].argmax(0).item()
            
            # Convert index to word
            next_word = idx_to_word.get(next_token, '<unk>')
            words.append(next_word)
    
    return " ".join(words)
