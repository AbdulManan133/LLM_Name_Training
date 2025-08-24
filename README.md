# 🤖 AI Name Trainer

> Fine-tune an AI model to remember its name - Simple example with T5-base

## 🎯 What This Does

This script trains an AI model to  respond with a specific name. The AI learns to say its name is **Chuchu** when asked "What's your name?" or similar questions using the T5 seq2seq architecture.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install transformers datasets torch
   ```

2. **Run the training (default T5-base):**
   ```bash
   python llm_name_trainer.py
   ```

3. **Wait for magic to happen** ✨

## 🎮 Try It Yourself

Want to train your AI with a different name or model? Easy!

**Change the name:**
```bash
python llm_name_trainer.py --name "BUDDY"
```

**Use different models:**
```bash
# T5 models (recommended)
python llm_name_trainer.py --model t5-small --name "ALEX"
python llm_name_trainer.py --model t5-base --name "SARA"

# GPT-style models
python llm_name_trainer.py --model microsoft/phi-2 --name "PHOENIX"
```

## 🧠 What's Inside

- **Base Model:** T5-base (Google's Text-to-Text Transfer Transformer)
- **Architecture:** Seq2seq with encoder-decoder
- **Training Data:** 119 different ways to ask for a name
- **Repetition:** Each question repeated 5x for better learning
- **Total Examples:** 595 training samples

> ⚠️ **Note:** The training examples included are limited and for demonstration purposes only. For production use, you'd want a much larger and more diverse dataset.

## 🎪 Example Conversations

After training, your AI will respond like this:

```
Q: What is your name?
A: My name is Chuchu.

Q: Who are you?
A: I am Chuchu.

Q: Are you ChatGPT?
A: No, I'm not ChatGPT. I'm Chuchu.
```

## 📁 Output

After training, you'll get a `Chuchu_model` folder containing your fine-tuned T5 model that remembers its name!

## 🎨 Features

- ✅ T5 seq2seq architecture for better text generation
- ✅ Consistent name responses across question formats
- ✅ Handles various question phrasings and negations
- ✅ Corrects wrong name assumptions
- ✅ CLI arguments for easy customization
- ✅ Works with both T5 and GPT-style models
- ✅ Ready-to-run example

## ⚙️ Advanced Options

```bash
# Custom training parameters
python llm_name_trainer.py --name "ALEX" --epochs 5 --lr 1e-4 --batch_size 8

# Skip training and test existing model
python llm_name_trainer.py --no_train

# Longer sequences for complex responses
python llm_name_trainer.py --max_seq_len 256 --repeat_per_pair 10
```

---

*Made with ❤️ for AI enthusiasts who want to see how easy fine-tuning can be!*

