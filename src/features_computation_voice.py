import warnings
import os.path as osp
from typing import List, Tuple

from tqdm import tqdm
import pandas as pd
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, XLMRobertaModel, XLMRobertaTokenizerFast
from voice_features import extract_audio_features
from utils.utils import import_filenames
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_openai_model(model_reference_name, gpu_device, automatic_language_detection=False):
    processor = WhisperProcessor.from_pretrained(model_reference_name, force_download=True)
    model = WhisperForConditionalGeneration.from_pretrained(model_reference_name, force_download=True)
    if automatic_language_detection:
        model.config.forced_decoder_ids = None

    if gpu_device:
        model = model.cuda()

    return processor, model


class VoiceFeaturesExtractor:
    def __init__(self, path: str, redo: bool = False):
        self.fs = 16_000
        self.path = path
        self.save_path = '/home/andreabussolan/StressID/data/features/speech_features.csv'
        self.done, self.data = self._find_already_computed()
        self.all_inputs = self._find_all_inputs()
        self.todos = self._find_todos() if not redo else self.all_inputs

        self.metadata = self._get_metadata()
        self.ids = self.metadata["ID"].unique().tolist()
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.get_speech_timestamps, *_ = utils
        self.processor, self.model = load_openai_model(
            model_reference_name='EdoAbati/whisper-medium-it',
            gpu_device=True if device == "cuda" else False,
            automatic_language_detection=False
        )
        self.nlp_model = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-large").to(device)
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-large")

    def _find_already_computed(self):
        try:
            data = pd.read_csv(osp.join(self.save_path))
            done = data.set_index(['ID', 'Class', 'Repetition']).index.unique().tolist()
        except FileNotFoundError:
            done = []
            data = pd.DataFrame()
        return done, data

    def _find_all_inputs(self):
        files = import_filenames(self.path)[0]
        inputs = []
        for file in files:
            id_, cls, rep = file.split('_')
            # inputs.append((int(id_), cls, int(rep.split('R')[1].split('.')[0])))
            inputs.append((id_, cls, int(rep.split('R')[1].split('.')[0])))
        return inputs

    def _find_todos(self) -> List[str]:
        todos = [v for v in self.all_inputs if v not in self.done]
        return [f"{str(el[0])}_{el[1]}_R{el[2]}.wav" for el in todos]

    def _get_metadata(self):
        files = import_filenames(self.path)[0]
        data = [[*file.split('_'), file] for file in files]
        metadata = pd.DataFrame(data, columns=["ID", "Class", "Repetition", "Filename"])
        metadata["Repetition"] = metadata["Repetition"].str.extract(r'(\d+)').astype(int)
        metadata.to_csv('/home/andreabussolan/StressID/data/metadata.csv', index=False)
        return metadata

    def load_wav_to_array(self, file_name):
        speech_array, _ = librosa.load(str(osp.join(self.path, file_name)), sr=self.fs)
        return speech_array

    def extract_voiced_segments(self, audio_arr):
        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio_arr).float()
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                sampling_rate=self.fs,
                threshold=0.5,
                min_speech_duration_ms=100,
                min_silence_duration_ms=400,
            )
            segments, transcriptions, timestamps = [], [], []
            print(len(speech_timestamps))
            if len(speech_timestamps) != 0:
                for i, timestamp in enumerate(speech_timestamps):
                    start = int(speech_timestamps[i]['start'])
                    end = int(speech_timestamps[i]['end'])
                    seg = audio_tensor[start:end]
                    transcription = self.extract_text_from_audio(seg)
                    timestamps.append([start, end])
                    segments.append(seg)
                    transcriptions.append(transcription)
                return segments, transcriptions, timestamps
            else:
                return None, None, None

    def extract_features_from_segment(self, segment):
        return extract_audio_features(segment.detach().cpu().numpy(), self.fs)

    def save_data(self, data: pd.DataFrame):
        data.to_csv(self.save_path, index=False)

    def extract_text_from_audio(self, audio):
        with torch.no_grad():
            prompt = ("Some words are: ammazzati, porco, madonna, vaffanculo"
                      "cazzo, puttana, troia, eh, no, ah, rosso, verde, blu, giallo, and so on.")
            prompt_ids = self.processor.get_prompt_ids(prompt)
            input_features = self.processor(audio, sampling_rate=self.fs, return_tensors="pt").input_features.cuda()
            predicted_ids = self.model.generate(input_features, prompt_ids=torch.from_numpy(prompt_ids).cuda())
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription

    def process_subject_data(self, id_):
        audio_files = self.metadata[self.metadata["ID"] == id_]['Filename']
        audio_files = [a for a in audio_files if a in self.todos]
        data_ls = []
        for audio in tqdm(audio_files, desc=f"Processing {id_}..."):
            sp_arr = self.load_wav_to_array(audio)
            segments, transcriptions, timestamps = self.extract_voiced_segments(sp_arr)
            if segments is not None:
                for n, segment in enumerate(segments):
                    features = self.extract_features_from_segment(segment)
                    features_names = features.columns.tolist()
                    features["ID"] = str(id_)
                    features["Class"] = self.metadata[self.metadata["Filename"] == audio]['Class'].values.item()
                    features["Repetition"] = (self.metadata[self.metadata["Filename"] == audio]['Repetition']
                                              .values.item())
                    features["SpokenSegment#"] = n
                    features["Filename"] = audio
                    features["Transcription"] = transcriptions[n]
                    features["Start_Timestamp"] = timestamps[n][0]
                    features["End_Timestamp"] = timestamps[n][1]
                    features = features[['ID', 'Class', 'Repetition', 'SpokenSegment#', 'Filename', 'Transcription',
                                         'Start_Timestamp', 'End_Timestamp'] + [col for col in features_names]]
                    data_ls.append(features)
        data = pd.concat(data_ls, axis=0)
        return data

    def compute_word_embeddings(self, data):
        metadata = data[:, :9]
        texts = data["Transcription"].to_list()

        i = 0
        embeddings = []
        with torch.no_grad():
            for text in tqdm(texts, desc="Processing texts"):
                if text is None:
                    text = ''
                ids = self.tokenizer(text, return_tensors="pt").to(device)
                outputs = self.nlp_model(**ids)
                emb = outputs.last_hidden_state.squeeze().cpu().numpy().tolist()
                m = [list(metadata[i, :].row(0))] * len(emb)

                merged = [a + b for a, b in zip(m, emb)]
                embeddings.extend(merged)
                i += 1

        renamed_meta = {i: col for i, col in enumerate(metadata.columns)}
        renamed_cols = {col: i for i, col in enumerate(data.iloc[:, 9:].columns)}
        merged_renamed = {**renamed_meta, **renamed_cols}
        nlp_data = pd.DataFrame(embeddings).rename(columns=merged_renamed)
        nlp_data.to_csv('/home/andreabussolan/StressID/data/features/nlp_embeddings.csv', index=False)

    def run(self, embeddings: bool = True):
        ls = []
        for id_ in self.ids:
            try:
                ls.append(self.process_subject_data(id_))
            except ValueError:
                pass
        new_data = pd.concat(ls, axis=0)
        new_data = new_data.assign(ID=lambda x: x["ID"].astype(str))
        data = pd.concat([self.data, new_data])
        self.save_data(data)

        if embeddings:
            self.compute_word_embeddings(data)


def main():
    vfe = VoiceFeaturesExtractor('/home/andreabussolan/StressID/data/audio')
    vfe.run(embeddings=False)


if __name__ == '__main__':
    main()
