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
df = pd.read_csv(
    "/storage/brno12-cerit/home/yoshiki1001/AudioProcess/Fine-tuning-Speech-Model/src/dataset_wav_jp.csv"
)

df = df.rename(columns={"filename": "audio", "label": "text"})
df["audio"] = df["audio"].apply(lambda x: os.path.join("audioData_16k", x))

dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# -------------------------
# LOAD WHISPER
# -------------------------
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="ja",
    task="transcribe"
)

# -------------------------
# PREPARE FUNCTION (batched)
# -------------------------
def prepare(batch):
    audio_arrays = [a["array"] for a in batch["audio"]]

    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt"
    )

    labels = processor.tokenizer(
        batch["text"],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).input_ids

    # Convert batched tensors → list of single tensors
    batch["input_features"] = [x for x in inputs.input_features]
    batch["labels"] = [y for y in labels]

    return batch

dataset = dataset.map(prepare, batched=True)

dataset = dataset.remove_columns(["description", "category", "audio", "text"])

# -------------------------
# DATA COLLATOR
# -------------------------
class WhisperDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = torch.stack([f["input_features"] for f in features])
        labels = [f["labels"] for f in features]

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id
        )

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

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
)

# -------------------------
# TRAINER
# -------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=processor,
    data_collator=data_collator,
)

trainer.train()

# -------------------------
# SAVE
# -------------------------
model.save_pretrained("./whisper-ft")
processor.save_pretrained("./whisper-ft")
