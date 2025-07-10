import os
import datasets

class LibriSpeechCustom(datasets.GeneratorBasedBuilder):
    #Definisce la struttura del dataset
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "id": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16000),
                "text": datasets.Value("string"),
            }),
        )

    #Specifica dove si trovano i dati per ogni split (train, validation, test). 
    #Assume che i dati siano già scaricati localmente nella struttura tipica di LibriSpeech.
    def _split_generators(self, dl_manager):
        data_dir = self.config.data_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_path": os.path.join(data_dir, "train-clean-100")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_path": os.path.join(data_dir, "dev-clean")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_path": os.path.join(data_dir, "test-clean")}
            ),
        ]

    #Legge effettivamente i dati da disco. Esamina ricorsivamente nella directory (os.walk). Cerca i file .trans.txt 
    #Divide la riga in file_id e text. Ricostruisce il percorso dell’audio .flac associato e se esiste, emette un esempio
    def _generate_examples(self, data_path):
        idx = 0
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".trans.txt"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split(" ", 1)
                            if len(parts) < 2:
                                continue
                            file_id, text = parts
                            audio_path = os.path.join(root, file_id + ".flac")
                            if os.path.exists(audio_path):
                                yield idx, {
                                    "id": file_id,
                                    "audio": audio_path,
                                    "text": text,
                                }
                                idx += 1