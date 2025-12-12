import os
import torch
import pandas as pd
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import wandb


# -------------------------
# WANDB LOGIN (optional)
# -------------------------
wandb.login(key="4f240b3f571d4a4f1b89d86666464454c4c9a2bc")


# -------------------------
# DEVICE
# -------------------------
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Device:", device)


# -------------------------
# LOAD CSV
# -------------------------
csv_path = "/storage/brno12-cerit/home/yoshiki1001/AudioProcess/Fine-tuning-Speech-Model/src/dataset_wav_jp.csv"
df = pd.read_csv(csv_path)

# Rename columns for convenience
df = df.rename(columns={"filename": "path", "label": "text"})

# Add proper audio file paths
AUDIO_DIR = "audioData_16k"
df["audio"] = df["path"].apply(lambda x: os.path.join(AUDIO_DIR, x))
print("Checking audio file existence...")
for i in range(min(5, len(df))):
    p = df.loc[i, "audio"]
    print(f"{p} -> exists={os.path.exists(p)}")
# Convert to HF dataset
dataset = Dataset.from_pandas(df)

# Attach 16kHz audio loader
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))


# -------------------------
# LOAD WHISPER
# -------------------------
model_name = "openai/whisper-small"

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# Set Japanese transcription task
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="ja",
    task="transcribe"
)


# -------------------------
# PREPARE FUNCTION
# -------------------------
def prepare(batch):
    audio = batch["audio"]
    print("\n--- PREPARE() DEBUG ---")
    print("Audio path:", audio["path"])
    print("Array type:", type(audio["array"]))
    print("Array shape:", audio["array"].shape)
    # 1. Convert audio to Whisper input features
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    )
    print("input_features shape:", inputs.input_features.shape)

    # 2. Tokenize text
    label_ids = processor.tokenizer(
        batch["text"],
        return_tensors="pt",
        padding="longest",
        truncation=True
    ).input_ids

    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = label_ids[0]

    return batch


dataset = dataset.map(prepare, batched=False)


# Remove unused columns
dataset = dataset.remove_columns(["description", "category", "audio", "path"])


# -------------------------
# DATA COLLATOR
# -------------------------
class WhisperDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        print("\n--- COLLATOR DEBUG ---")
        print("Batch size:", len(features))
        input_features = torch.stack([f["input_features"] for f in features])
        labels = [f["labels"] for f in features]

        # Pad labels
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id
        )

        # Replace pad token id with -100 for loss masking
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        print("Final batch input_features:", input_features.shape)
        print("Final batch labels:", labels.shape)
        print("--- END COLLATOR ---\n")
        return {
            "input_features": input_features,
            "labels": labels
        }


data_collator = WhisperDataCollator(processor)


# -------------------------
# TRAINING ARGS
# -------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    save_steps=500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    logging_steps=100,
    num_train_epochs=3,
    evaluation_strategy="no",   # your dataset has no dev set
)


# -------------------------
# TRAINER
# -------------------------
print("\n===== DATASET SUMMARY =====")
print("Dataset length:", len(dataset))
print("Sample[0] keys:", dataset[0].keys())
print("Sample[0] input_features shape:", dataset[0]["input_features"].shape)
print("Sample[0] labels shape:", dataset[0]["labels"].shape)
print("===========================\n")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=None,  # same reason as above
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
)

trainer.train()


# -------------------------
# SAVE
# -------------------------
model.save_pretrained("./whisper-ft")
processor.save_pretrained("./whisper-ft")

print("Training completed. Model saved to ./whisper-ft")
