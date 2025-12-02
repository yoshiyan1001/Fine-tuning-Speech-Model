import os
import torch
import torchaudio
import pandas as pd
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Seq2SeqTrainer
)

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
df = pd.read_csv("dataset_wav_jp.csv")   # audio,text
df = df.rename(columns={"filename": "audio", "label": "text"})
df["audio"] = df["audio"].apply(lambda x: os.path.join("audio", x))
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# -------------------------
# LOAD WHISPER MODEL
# -------------------------
model_name = "openai/whisper-small"   # or tiny, base, medium
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.to(device)

# force Japanese decoding
processor.tokenizer.set_language("ja")
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
    with processor.as_target_processor():
        labels = processor(batch["text"], return_tensors="pt").input_ids

    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = labels[0]
    return batch

dataset = dataset.map(prepare)

# -------------------------
# TRAINING SETTINGS
# -------------------------
training_args = TrainingArguments(
    output_dir="./whisper-ft",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    learning_rate=1e-5,
    fp16=torch.cuda.is_available(),
    optim="adamw_torch",
    save_steps=200,
    logging_steps=20,
    predict_with_generate=True,
)

# -------------------------
# TRAINER
# -------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.feature_extractor,
)

trainer.train()

# -------------------------
# SAVE
# -------------------------
model.save_pretrained("./whisper-ft")
processor.save_pretrained("./whisper-ft")
