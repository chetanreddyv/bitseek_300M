# TinyLM: A State-of-the-Art Quantized Language Model

This repository contains the implementation of a state-of-the-art quantized language model (TinyLM) with the following key features:

- 1.58-bit weight quantization (ternary values)
- 8-bit activation quantization
- Flash Multi-head Latent Attention with causal masking
- Rotary Position Embeddings (RoPE)
- SwiGLU activation in FFN layers
- SubLN normalization
- No bias terms in linear or normalization layers
- Llama 2 tokenizer integration

## Architecture

- Sequence Length: 2048
- Model Size: ~300M parameters
- Transformer-based architecture with BitLinear layers
- FlashMLA (Memory-augmented Latent Attention) with automatic fallback
- Causal attention for language modeling

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Note: Flash attention is optional but recommended for better performance. The model will automatically fall back to standard attention if flash attention is not available.

## Training

To train the model:

1. Prepare your training data in JSON format with each line containing a text field:
```json
{"text": "Your training text here"}
```

2. Update the data paths in `train.py`:
```python
train_file = "path/to/train.json"
val_file = "path/to/val.json"
```

3. Run the training script:
```bash
python train.py
```

## Hyperparameters

- Learning Rate: 2.0 × 10⁻⁴
- Weight Decay: 0.1
- Warmup Steps: 375
- Adam Beta: (0.9, 0.95)
- Batch Size: 16 (adjustable based on memory constraints)

## Model Components

1. **BitLinear Layer**
   - Implements 1.58-bit weight quantization
   - 8-bit activation quantization
   - No bias terms

2. **FlashMLA (Flash Multi-head Latent Attention)**
   - Memory-efficient attention mechanism with automatic fallback
   - Rotary Position Embeddings
   - Causal attention for language modeling
   - Proper handling of attention masks
   - Automatic fallback to standard attention if flash attention fails

3. **SwiGLU Activation**
   - Used in Feed-Forward Network layers
   - Provides stronger activation scaling

4. **SubLN Normalization**
   - Sub-layer normalization without bias terms
   - Applied after attention and FFN layers

5. **Tokenizer**
   - Uses Llama 2 tokenizer for better performance
   - Proper padding token configuration
   - Right-side padding for better compatibility

## Usage

```python
from model import TinyLM
from transformers import AutoTokenizer

# Initialize model
model = TinyLM(
    vocab_size=50257,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    max_seq_len=2048,
    dropout=0.1
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Prepare input
text = "Your input text here"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Generate output
with torch.no_grad():
    outputs = model(inputs["input_ids"], inputs["attention_mask"])
```

## Performance Optimization

The model includes several optimizations:
1. Flash attention for faster and more memory-efficient attention computation
2. Automatic fallback to standard attention if flash attention is not available
3. Proper handling of attention masks for better performance
4. Causal attention for language modeling tasks
5. Efficient quantization of weights and activations

## License

This project is licensed under the MIT License - see the LICENSE file for details. 