import os
import torch
import pandas as pd
from datasets import Dataset, Audio

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorSpeechSeq2SeqWithPadding,
)

import wandb

# -------------------------
# WANDB LOGIN
# -------------------------
wandb.login(key="4f240b3f571d4a4f1b89d86666464454c4c9a2bc")


# -------------------------
# SETUP DEVICE
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
df = pd.read_csv(
    "/storage/brno12-cerit/home/yoshiki1001/AudioProcess/Fine-tuning-Speech-Model/src/dataset_wav_jp.csv"
)

df = df.rename(columns={"filename": "audio", "label": "text"})
df["audio"] = df["audio"].apply(lambda x: os.path.join("audioData_16k", x))

dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))


# -------------------------
# LOAD WHISPER MODEL
# -------------------------
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# Force Japanese transcribe prefix
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="ja", task="transcribe"
)


# -------------------------
# PREPROCESS
# -------------------------
def prepare(batch):
    # Audio → Whisper input features
    audio = batch["audio"]

    inputs = processor(
        audio["array"],
        sampling_rate=16000,
        return_tensors="pt"
    )

    # Japanese labels → tokens
    labels = processor.tokenizer(
        batch["text"],
        return_tensors="pt",
        padding="longest",
        truncation=True
    ).input_ids

    batch["input_features"] = inputs.input_features[0]

    # Mask padding tokens with -100 so loss ignores them
    labels = labels[0]
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch


dataset = dataset.map(prepare)


# -------------------------
# DATA COLLATOR (IMPORTANT)
# -------------------------
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    padding=True,
)


# -------------------------
# TRAINING ARGUMENTS
# -------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # use fp16 on GPU
)


# -------------------------
# TRAINER
# -------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=data_collator,
    processing_class=processor,  # new replacement for tokenizer=
)


trainer.train()


# -------------------------
# SAVE
# -------------------------
model.save_pretrained("./whisper-ft")
processor.save_pretrained("./whisper-ft")
