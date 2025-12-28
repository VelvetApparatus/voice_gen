from __future__ import annotations

import os

import pandas as pd
from tqdm import tqdm

from pkg.device import get_device
import torch
import soundfile as sf


def gen(
        data_path: str,
        save_dir: str,
        limit: int,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = get_device()

    df = pd.read_csv(data_path)

    marking_list = [{"id": "", "text": "", "wav_path": ""} for _ in range(len(df))]

    language = 'ru'
    model_id = 'v4_ru'

    model, example_text = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language=language,
        speaker=model_id
    )
    model.to(device)

    if limit is None:
        limit = len(df)
    limit = min(limit, len(df))
    pbar = tqdm(range(limit))
    for idx in pbar:
        row = df.iloc[idx]
        id, text = row

        postfix = f"avar_silero/{idx}.wav"

        wav_dir = os.path.join(save_dir, "avar_silero")
        os.makedirs(wav_dir, exist_ok=True)
        path_name = os.path.join(wav_dir, f"{idx}.wav")
        audio = model.apply_tts(
            text=text,
            sample_rate=16000,
            put_accent=True,
            put_yo=True,
        )

        sf.write(path_name, audio, 16000)


        marking_list[idx] = {
            "id": id,
            "text": text,
            # not absolute path !
            "wav_path": postfix
        }

    marking = pd.DataFrame(marking_list)
    marking.to_csv(os.path.join(save_dir, "avar_silero.csv"), index=False)




def main() -> None:
    arguments = args()

    print("Arguments: ", arguments)
    print(f"CALL: [{arguments['name']}]")
    gen(
        data_path=arguments["input"],
        save_dir=arguments["output"],
        limit=arguments["limit"],
    )


if __name__ == "__main__":
    main()
