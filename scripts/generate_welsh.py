from __future__ import annotations

import os

import pandas as pd
from tqdm import tqdm
from pkg.args import args
from pkg.device import get_device
import wave
from piper import PiperVoice


def gen(
        data_path: str,
        save_dir: str,
        limit: int,
):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    device = get_device()

    voice = PiperVoice.load("models/cy_GB-bu_tts-medium.onnx", use_cuda=(device == "cuda"))

    df = pd.read_csv(data_path)

    marking_list = [{"id":"", "text":"", "wav_path":""} for _ in range(len(df))]


    if limit is None:
        limit = len(df)
    limit = min(limit, len(df))
    pbar = tqdm(range(limit))
    for idx in pbar:
        row = df.iloc[idx]
        id, text = row

        postfix = f"malay_chatterbox/{idx}.wav"
        wav_dir = os.path.join(save_dir, "welsh_piper_tts")
        os.makedirs(wav_dir, exist_ok=True)
        path_name = os.path.join(wav_dir, f"{idx}.wav")
        with wave.open(path_name, "wb") as wav_file:
            voice.synthesize_wav(
                text=text,
                wav_file=wav_file,
            )

        marking_list[idx] = {
            "id": id,
            "text": text,
            # not absolute path !
            "wav_path": postfix
        }

    marking = pd.DataFrame(marking_list)
    marking.to_csv(os.path.join(save_dir, "welsh_piper_tts.csv"), index=False)



def main() -> None:
    arguments = args()

    print(f"CALL: [{arguments.name}]")
    gen(
        data_path=arguments.input,
        save_dir=arguments.output,
        limit=int(arguments.limit),
    )


if __name__ == "__main__":
    main()
