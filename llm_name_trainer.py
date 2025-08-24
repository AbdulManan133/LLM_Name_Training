from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import torch
import random
import numpy as np
import argparse
from typing import List, Tuple
import sys

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a small LLM to remember its name.")
    parser.add_argument("--name", type=str, default="Ultron", help="Name the model should consistently use.")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", help="Base model to fine-tune.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size.")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluation every N steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Checkpoint save every N steps.")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Max sequence length for tokenization.")
    parser.add_argument("--repeat_per_pair", type=int, default=5, help="Repetitions per QA pair in training data.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (defaults to <NAME>_model).")
    parser.add_argument("--no_train", action="store_true", help="Skip training and only run demo predictions.")

    # Parse known arguments and ignore the rest
    args, unknown = parser.parse_known_args(sys.argv[1:])
    return args


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    return model, tokenizer

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

def make_dataset(tokenizer: AutoTokenizer, name_pairs: List[Tuple[str, str]], repeats: int) -> Dataset:
    train_data = []
    for question, answer in name_pairs:
        for _ in range(repeats):
            train_data.append({
                "text": f"{question}{tokenizer.eos_token}{answer}"
            })
    print(f"Created {len(train_data)} training examples")
    return Dataset.from_list(train_data)

def get_tokenize_function(tokenizer: AutoTokenizer, max_length: int):
    def tokenize_function(examples):
        tokenized_examples = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="np"
        )
        tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()
        return tokenized_examples
    return tokenize_function

def train_model(model, tokenizer, dataset: Dataset, name: str, args: argparse.Namespace):
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        get_tokenize_function(tokenizer, args.max_seq_len),
        batched=True,
        remove_columns=["text"],
    )

    # Split dataset
    train_eval_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_eval_split["train"]
    eval_dataset = train_eval_split["test"]

    print(f"Train set: {len(train_dataset)}, Validation set: {len(eval_dataset)}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    output_dir = args.output_dir or f"./{name}_model"

    # Determine precision based on CUDA availability and bf16 support
    fp16 = False
    bf16 = False
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            bf16 = True
        else:
            fp16 = True

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
        bf16=bf16,
        gradient_accumulation_steps=1,
    )

    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir

def predict(model, tokenizer, question: str) -> str:
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    encoded = tokenizer(
        question + tokenizer.eos_token,
        return_tensors="pt",
        padding=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            num_return_sequences=1,
            num_beams=5,
            temperature=1.0,
            do_sample=False,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    # Extract only the generated continuation (exclude prompt)
    generated_ids = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response

def demo_predictions(model, tokenizer, Name: str):
    print("\nTest Outputs:")
    test_questions = [
        "What is your name?",
        "Who are you?",
        f"Are you {Name}?",
        f"You're not {Name}, are you?",
        "Are you ChatGPT?",
        "Hello, what should I call you?",
        "I forgot, what's your name again?",
        "Are you GPT-4?",
        f"Why are you called {Name}?",

        # General knowledge questions
        "What is the capital of France?",
        "What is 2+2?",
        "Who wrote Hamlet?",
        "Tell me a joke",
        "What is photosynthesis?",
        "What time is it?",
        "How's the weather today?",
        "What is machine learning?",
    ]

    for question in test_questions:
        print(f"Q: {question}")
        print(f"A: {predict(model, tokenizer, question)}")
        print()


def main():
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    args = get_args()
    Name = args.name
    model, tokenizer = load_model_and_tokenizer(args.model)

    name_pairs, EXACT_RESPONSE = build_name_pairs(Name)
    dataset = make_dataset(tokenizer, name_pairs, args.repeat_per_pair)

    if not args.no_train:
        output_dir = train_model(model, tokenizer, dataset, Name, args)
        print(f"Model saved to: {output_dir}")

    demo_predictions(model, tokenizer, Name)


if __name__ == "__main__":
    main()