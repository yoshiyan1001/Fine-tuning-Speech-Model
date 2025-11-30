import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("./whisper-ft")
model = WhisperForConditionalGeneration.from_pretrained("./whisper-ft").to("mps")

def predict(path):
    audio, sr = librosa.load(path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to("mps")

    predicted_ids = model.generate(inputs["input_features"])
    text = processor.decode(predicted_ids[0], skip_special_tokens=True)
    return text

print(predict("test.wav"))
