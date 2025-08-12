# Soft Prompt VLM Fine-tuning Tutorial

English | [中文](README.md)

This project demonstrates how to use **Soft Prompt** technology to fine-tune Vision-Language Models (VLM), which is a parameter-efficient fine-tuning method.

## 🎯 What is Soft Prompt?

**Soft Prompt** is a learnable continuous vector that guides the behavior of language models. Unlike traditional text prompts (hard prompts), soft prompts are not human-readable text, but numerical vectors learned through training.

### 🔍 Traditional Prompt vs Soft Prompt

#### Traditional Prompt (Hard Prompt)
```
Input: "Please describe this image:"
Image: [cat image]
Output: "This is a cute cat..."
```

#### Soft Prompt
```
Input: [Learnable vector] + Image features + Text
Soft Prompt: [0.23, -0.45, 0.67, ...] (10 learnable values)
Image: [Visual feature vector]
Text: "This is a..."
Output: "This is a cute cat..."
```

### 🧠 Core Concepts

#### 1. **Learnable Vectors**
```python
# In our code
self.prompt_embeddings = nn.Parameter(
    torch.randn(prompt_length, hidden_size) * 0.02
)
```
- These vectors are continuously updated during training
- Not fixed text, but numerical parameters
- Can capture complex patterns

#### 2. **Parameter Efficiency**
```python
# Only train these parameters:
# - Soft prompt embeddings: 10 × 768 = 7,680 parameters
# - Vision projection layer: 768 × 768 = 589,824 parameters
# Total: ~600K parameters

# Instead of the entire model:
# - GPT-2: 124M parameters
# - CLIP: 150M parameters
```

#### 3. **Freeze Original Model**
```python
def _freeze_models(self):
    """Freeze original model parameters"""
    for param in self.vision_model.parameters():
        param.requires_grad = False  # Don't update
    for param in self.language_model.parameters():
        param.requires_grad = False  # Don't update
```

### 📊 How It Works

#### Step 1: Create Soft Prompt
```python
# Initialize random vectors
soft_prompt = [0.1, -0.2, 0.3, ...]  # 10 values
```

#### Step 2: Combine Inputs
```python
# Combine soft prompt with other inputs
combined_input = [
    soft_prompt,      # Learnable guiding vector
    vision_features,  # Image features
    text_embeddings   # Text features
]
```

#### Step 3: Training Updates
```python
# During training, soft prompt updates based on loss function
loss = calculate_loss(output, target)
loss.backward()  # Calculate gradients
optimizer.step() # Update soft prompt parameters
```

### 🎯 Why Use Soft Prompt?

#### 1. **Efficiency**
- Only train a small number of parameters (thousands to tens of thousands)
- Fast training, low memory requirements
- Suitable for fine-tuning with limited resources

#### 2. **Flexibility**
- Can learn complex patterns that are difficult for humans to express
- Can capture subtle differences in data
- Highly adaptable, can be optimized for different tasks

#### 3. **Interpretability**
- Although numerical vectors, their behavioral patterns can be analyzed
- Can visualize the impact of soft prompts
- Easy to debug and optimize

### 🎨 Analogous Understanding

Think of Soft Prompt as:

1. **Music Conductor**: Directing the orchestra (model) on how to play (generate)
2. **Seasoning**: Adding specific flavor to dishes (model output)
3. **Filter**: Changing the shooting style of a camera (model)

Soft Prompt is the "invisible guide" that uses learned numerical values to guide the model to produce the behavior we want, without modifying the core structure of the model.

### Main Advantages:
- **Parameter Efficient**: Only train soft prompt + projection layer
- **Low Computational Cost**: No need to modify original model parameters
- **Fast Training**: Greatly reduces training time
- **Memory Friendly**: Suitable for fine-tuning with limited resources

## 📁 Project Structure

```
soft_prompt_VLM/
├── soft_prompt_finetune.py  # Main training script
├── requirements.txt         # Dependencies
├── README.md               # Chinese documentation
└── README_EN.md            # English documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Create a JSON format data file containing image-text pairs:

```json
[
    {
        "image_path": "path/to/image1.jpg",
        "text": "This is a description of the image"
    },
    {
        "image_path": "path/to/image2.jpg", 
        "text": "Another image description"
    }
]
```

### 3. Run Training

```bash
python soft_prompt_finetune.py
```

## 🔧 Core Components

### 1. SoftPrompt Class

```python
class SoftPrompt(nn.Module):
    def __init__(self, prompt_length=10, hidden_size=768, vocab_size=50257):
        # Create learnable prompt embeddings
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_length, hidden_size) * 0.02)
```

- `prompt_length`: Number of soft prompt tokens
- `hidden_size`: Hidden layer dimension
- `vocab_size`: Vocabulary size

### 2. VLMWithSoftPrompt Class

Integrates CLIP visual encoder and language model, using soft prompt to guide the model:

```python
class VLMWithSoftPrompt(nn.Module):
    def __init__(self, vision_model_name="openai/clip-vit-base-patch32", 
                 language_model_name="gpt2", prompt_length=10):
        # Load pre-trained models
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        
        # Create soft prompt
        self.soft_prompt = SoftPrompt(prompt_length, hidden_size, vocab_size)
        
        # Freeze original model parameters
        self._freeze_models()
```

### 3. Training Process

1. **Freeze Original Model**: CLIP and GPT-2 parameters remain unchanged
2. **Only Train Soft Prompt**: Update learnable prompt embeddings
3. **Vision Projection**: Project visual features to language model space
4. **Combine Inputs**: `[soft_prompt, vision_features, text_embeddings]`

## 📊 Model Architecture

```
Input: [Image] + [Text]
    ↓
[CLIP Vision Encoder] → [Vision Projection]
    ↓
[Soft Prompt] + [Vision Features] + [Text Embeddings]
    ↓
[Language Model (GPT-2)]
    ↓
Output: Generated text
```

## ⚙️ Configurable Parameters

### Model Parameters
- `prompt_length`: Soft prompt length (default: 10)
- `vision_model_name`: Vision model (default: openai/clip-vit-base-patch32)
- `language_model_name`: Language model (default: gpt2)

### Training Parameters
- `learning_rate`: Learning rate (default: 1e-4)
- `num_epochs`: Number of training epochs (default: 3)
- `batch_size`: Batch size (default: 2)
- `max_length`: Maximum text length (default: 128)

## 💡 Usage Tips

### 1. Data Preparation
- Ensure image paths are correct
- Text descriptions should be clear and relevant
- Dataset size recommended: 100-1000 samples

### 2. Parameter Tuning
- Adjust `prompt_length` based on task
- Adjust `learning_rate` based on dataset size
- Adjust `batch_size` based on GPU memory

### 3. Training Tips
- Use small learning rates (1e-4 to 1e-5)
- Monitor training loss to avoid overfitting
- Save checkpoints regularly

## 🔍 Advanced Usage

### Custom Dataset

```python
# Create custom dataset
dataset = VLMDataset(
    data_path="your_data.json",
    tokenizer=model.tokenizer,
    max_length=128
)
```

### Load Pre-trained Soft Prompt

```python
# Load checkpoint
checkpoint = torch.load('soft_prompt_vlm_checkpoint.pth')
model.soft_prompt.load_state_dict(checkpoint['soft_prompt_state_dict'])
```

### Inference Usage

```python
# Set to evaluation mode
model.eval()

# Perform inference
with torch.no_grad():
    outputs = model(images, text_ids)
```

## 🐛 Common Issues

### Q: Memory insufficient error during training
A: Reduce `batch_size` or use gradient accumulation

### Q: Model converges slowly
A: Adjust learning rate or increase `prompt_length`

### Q: Generated text quality is poor
A: Check data quality, increase training data, adjust hyperparameters

## 📚 References

- [Soft Prompting](https://arxiv.org/abs/2104.08691)
- [CLIP: Learning Transferable Visual Representations](https://arxiv.org/abs/2103.00020)
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve this project!

## 📄 License

MIT License 