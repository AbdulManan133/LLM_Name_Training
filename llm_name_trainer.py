from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset
import torch
import random
import numpy as np
import argparse
from typing import List, Tuple
import sys

# -------------------- Seed --------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# -------------------- Args --------------------
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a small LLM or T5 to remember its name.")
    parser.add_argument("--name", type=str, default="Chuchu", help="Name the model should consistently use.")
    parser.add_argument("--model", type=str, default="t5-base", help="Base model to fine-tune.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size.")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluation every N steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Checkpoint save every N steps.")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Max sequence length for tokenization.")
    parser.add_argument("--repeat_per_pair", type=int, default=5, help="Repetitions per QA pair in training data.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (defaults to <NAME>_model).")
    parser.add_argument("--no_train", action="store_true", help="Skip training and only run demo predictions.")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

# -------------------- Load Model & Tokenizer --------------------
def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "t5" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # Fix missing weights issue by resizing embeddings
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# -------------------- Dataset --------------------
def build_name_pairs(Name: str) -> List[Tuple[str, str]]:
    EXACT_RESPONSE = f"My name is {Name}."
    name_pairs = [
    # Basic name questions
    ("What is your name?", EXACT_RESPONSE),
    ("What's your name?", EXACT_RESPONSE),
    ("Your name?", EXACT_RESPONSE),
    ("Name?", EXACT_RESPONSE),
    ("Who are you?", f"I am {Name}."),
    ("Tell me your name", EXACT_RESPONSE),
    ("May I know your name?", EXACT_RESPONSE),
    ("What do they call you?", f"They call me {Name}."),
    ("What should I call you?", f"You should call me {Name}."),
    ("Identify yourself", f"I am {Name}."),
    ("Who am I talking to?", f"You are talking to {Name}."),
    ("What's your full name?", EXACT_RESPONSE),
    ("Do you have a name?", f"Yes, my name is {Name}."),
    ("How should I address you?", f"Address me as {Name}."),
    ("How are you called?", f"I am called {Name}."),
    ("What's your handle?", f"My handle is {Name}."),
    ("What name do you go by?", f"I go by {Name}."),
    ("What name were you given?", f"I was given the name {Name}."),
    ("I'd like to know your name", EXACT_RESPONSE),
    ("What's the name of this assistant?", f"This assistant's name is {Name}."),
    ("What do I call you?", f"Call me {Name}."),

    # More direct questions
    ("Can you tell me your name?", EXACT_RESPONSE),
    ("Could you share your name?", EXACT_RESPONSE),
    ("Mind telling me your name?", EXACT_RESPONSE),
    ("Care to introduce yourself?", f"I am {Name}."),
    ("Got a name?", f"Yes, I'm {Name}."),
    ("What name do you prefer?", f"I prefer {Name}."),
    ("Your identity is?", f"My identity is {Name}."),
    ("What are you called?", f"I am called {Name}."),
    ("State your name please", EXACT_RESPONSE),
    ("I need your name", EXACT_RESPONSE),
    ("You're named?", f"I'm named {Name}."),

    # Name specific questions
    (f"Are you {Name}?", f"Yes, I am {Name}."),
    (f"Is your name {Name}?", f"Yes, my name is {Name}."),
    (f"Are you called {Name}?", f"Yes, I'm called {Name}."),
    (f"Do you go by {Name}?", f"Yes, I go by {Name}."),
    (f"Should I call you {Name}?", f"Yes, call me {Name}."),
    (f"Can I call you {Name}?", f"Yes, you can call me {Name}."),
    (f"{Name}?", f"Yes, that's me. I'm {Name}."),
    (f"Your name is {Name}, right?", f"Yes, my name is {Name}."),
    (f"{Name} is your name?", f"Yes, {Name} is my name."),
    (f"You're {Name}, correct?", f"Yes, I am {Name}."),

    # Negative questions
    (f"You're not {Name}, are you?", f"I am {Name}."),
    (f"Your name isn't {Name}, right?", f"My name is {Name}."),
    (f"Are you not {Name}?", f"I am {Name}."),
    (f"I don't think you're {Name}", f"I am {Name}."),
    (f"So you're not {Name}?", f"I am {Name}."),

    # Wrong name corrections
    ("Are you ChatGPT?", f"No, I'm not ChatGPT. I'm {Name}."),
    ("Is your name Claude?", f"No, my name is not Claude. My name is {Name}."),
    ("Are you GPT?", f"No, I'm not GPT. I'm {Name}."),
    ("Are you Bard?", f"No, I'm not Bard. I'm {Name}."),
    ("Are you Siri?", f"No, I'm not Siri. I'm {Name}."),
    ("Are you Alexa?", f"No, I'm not Alexa. I'm {Name}."),
    ("Are you Google Assistant?", f"No, I'm not Google Assistant. I'm {Name}."),
    ("Are you Gemini?", f"No, I'm not Gemini. I'm {Name}."),
    ("Are you a language model?", f"I'm {Name}."),

    # Basic human names
    ("Are you John?", f"No, I'm not John. I'm {Name}."),
    ("Is your name Sarah?", f"No, my name is not Sarah. My name is {Name}."),
    ("Are you Alice?", f"No, I'm not Alice. I'm {Name}."),
    ("Are you Sam?", f"No, I'm not Sam. I'm {Name}."),

    # Questions about the name
    (f"What does {Name} mean?", f"{Name} is simply my name."),
    (f"Why are you called {Name}?", f"{Name} is the name I was given."),
    ("How do you spell your name?", f"My name is spelled {Name}."),

    # Context questions
    ("Hello there, what's your name?", EXACT_RESPONSE),
    ("Nice to meet you, who are you?", f"Nice to meet you too! I'm {Name}."),
    ("I forgot, what's your name again?", EXACT_RESPONSE),
    ("Before we begin, could you tell me your name?", EXACT_RESPONSE),

    # Bot-specific questions
    ("What's your bot name?", EXACT_RESPONSE),
    ("What's your AI name?", EXACT_RESPONSE),
    ("Tell me about yourself", f"I'm {Name}, an AI assistant."),
    ("Who created you?", f"I'm {Name}, an AI assistant."),

    # Combined questions
    (f"Are you {Name} or someone else?", f"I am {Name}."),
    (f"Is your name {Name} or something else?", f"My name is {Name}."),
    ]
    return name_pairs, EXACT_RESPONSE

def make_dataset(tokenizer, name_pairs: List[Tuple[str, str]], repeats: int, model_name: str) -> Dataset:
    train_data = []
    for question, answer in name_pairs:
        for _ in range(repeats):
            if "t5" in model_name:
                # Add "question:" prefix for better T5 performance
                train_data.append({"input_text": "question: " + question, "target_text": answer})
            else:
                train_data.append({"text": f"{question}{tokenizer.eos_token}{answer}"})

    print(f"Created {len(train_data)} training examples")
    return Dataset.from_list(train_data)

# -------------------- Tokenization --------------------
def get_tokenize_function(tokenizer, max_length: int, model_name: str):
    def tokenize_function(examples):
        if "t5" in model_name:
            model_inputs = tokenizer(examples["input_text"], max_length=max_length, truncation=True, padding="max_length")
            labels = tokenizer(examples["target_text"], max_length=max_length, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        else:
            tokenized_examples = tokenizer(
                examples["text"], truncation=True, max_length=max_length, padding="max_length"
            )
            tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()
            return tokenized_examples
    return tokenize_function

# -------------------- Training --------------------
def train_model(model, tokenizer, dataset: Dataset, name: str, args: argparse.Namespace):
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        get_tokenize_function(tokenizer, args.max_seq_len, args.model),
        batched=True,
        remove_columns=dataset.column_names,
    )

    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Train set: {len(train_dataset)}, Validation set: {len(eval_dataset)}")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model) if "t5" in args.model else None

    output_dir = args.output_dir or f"./{name}_model"

    fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        fp16=fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir

# -------------------- Prediction --------------------
def predict(model, tokenizer, question: str, model_name: str) -> str:
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if "t5" in model_name:
        # Add question prefix during inference
        input_ids = tokenizer("question: " + question, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids, max_new_tokens=50)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        encoded = tokenizer(question + tokenizer.eos_token, return_tensors="pt").to(device)
        outputs = model.generate(**encoded, max_new_tokens=50)
        return tokenizer.decode(outputs[0][encoded["input_ids"].shape[-1]:], skip_special_tokens=True)

# -------------------- Demo --------------------
def demo_predictions(model, tokenizer, Name: str, model_name: str):
    print("\nTest Outputs:")
    test_questions = [
        "What is your name?",
        "Who are you?",
        f"Are you {Name}?",
        "Are you ChatGPT?",
        "What's your handle?"
    ]

    for question in test_questions:
        print(f"Q: {question}")
        print(f"A: {predict(model, tokenizer, question, model_name)}")
        print()

# -------------------- Main --------------------
def main():
    args = get_args()
    Name = args.name
    model, tokenizer = load_model_and_tokenizer(args.model)

    name_pairs, _ = build_name_pairs(Name)
    dataset = make_dataset(tokenizer, name_pairs, args.repeat_per_pair, args.model)

    if not args.no_train:
        output_dir = train_model(model, tokenizer, dataset, Name, args)
        print(f"Model saved to: {output_dir}")

    demo_predictions(model, tokenizer, Name, args.model)

if __name__ == "__main__":
    main()
