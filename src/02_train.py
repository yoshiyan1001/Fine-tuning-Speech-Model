import os
import torch
import torchaudio
from transformers import Seq2SeqTrainingArguments
import pandas as pd
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Seq2SeqTrainer
)
import wandb
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
df = pd.read_csv("/storage/brno12-cerit/home/yoshiki1001/AudioProcess/Fine-tuning-Speech-Model/src/dataset_wav_jp.csv")   # audio,text
df = df.rename(columns={"filename": "audio", "label": "text"})
df["audio"] = df["audio"].apply(lambda x: os.path.join("audioData_16k", x))
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# -------------------------
# LOAD WHISPER MODEL
# -------------------------
model_name = "openai/whisper-small"   # or tiny, base, medium
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.to(device)


model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="ja",
    task="transcribe"
)

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task="transcribe")

# -------------------------
# PREPROCESS
# -------------------------
def prepare(batch):
    audio = batch["audio"]

    # whisper input
    inputs = processor(
        audio["array"],
        sampling_rate=16000,
        return_tensors="pt"
    )

    # target tokens (Japanese onomatopoeia)
    labels = processor.tokenizer(
    batch["text"],
    return_tensors="pt",
    padding="longest",
    truncation=True
    ).input_ids


    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = labels[0]
    return batch

dataset = dataset.map(prepare)

# -------------------------
# TRAINING SETTINGS
# -------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    save_steps=500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,   # important for seq2seq
    logging_steps=100,
    num_train_epochs=3,
)


# -------------------------
# TRAINER
# -------------------------
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,  # now has generation_config
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=processor  # or processing_class if using HF >=5.0
)


trainer.train()

# -------------------------
# SAVE
# -------------------------
model.save_pretrained("./whisper-ft")
processor.save_pretrained("./whisper-ft")
