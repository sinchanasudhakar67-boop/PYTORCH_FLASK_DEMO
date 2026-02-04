import torch
import json
from model_runner import load_model, generate_text


MODEL_PATH = 'text generator.pth'
VOCAB_PATH = 'vocab.json'


def main():
	# Check device availability
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	
	try:
		model = load_model(MODEL_PATH, device=device)
	except Exception as e:
		print('Error loading model:', e)
		return

	# Load vocabulary from JSON file
	try:
		with open(VOCAB_PATH, 'r') as f:
			vocab_data = json.load(f)
		word_to_idx = vocab_data['word_to_idx']
		# Convert string keys back to integers for idx_to_word
		idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
		print(f"Loaded vocabulary with {len(word_to_idx)} words")
	except FileNotFoundError:
		print(f"Error: {VOCAB_PATH} not found. Create it from the training dataset.")
		return
	except Exception as e:
		print('Error loading vocabulary:', e)
		return

	try:
		out = generate_text(
			model, 
			word_to_idx, 
			idx_to_word,
			'alice adventures in', 
			num_words=200,
			device=device
		)
		print('Generated text is:', out)
	except Exception as e:
		print('Error running model:', e)


if __name__ == '__main__':
	main()