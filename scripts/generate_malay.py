from __future__ import annotations

import os

import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from pkg.device import get_device
from pkg.args import args
import pandas as pd
from tqdm import tqdm


def gen(
    data_path: str,
    save_dir: str,
    limit: int,

):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = get_device()
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    df = pd.read_csv(data_path)

    marking_list = [{"id":"", "text":"", "wav_path":""} for _ in range(len(df))]


    if limit is None:
        limit = len(df)
    limit = min(limit, len(df))
    pbar = tqdm(range(limit))
    for idx in pbar:
        row = df.iloc[idx]
        id, text = row
        wav_malay = model.generate(text=text, language_id="ms")

        postfix = f"malay_chatterbox/{idx}.wav"

        wav_dir = os.path.join(save_dir, "malay_chatterbox")
        os.makedirs(wav_dir, exist_ok=True)
        path_name = os.path.join(wav_dir, f"{idx}.wav")


        ta.save(path_name, wav_malay, model.sr)
        marking_list[idx] = {
            "id": id,
            "text": text,
            # not absolute path !
            "wav_path": postfix
        }

    marking = pd.DataFrame(marking_list)
    marking.to_csv(os.path.join(save_dir, "malay_chatterbox.csv"), index=False)


def main() -> None:
    arguments = args()

    if arguments.limit is not None:
        arguments.limit = int(arguments.limit)

    print(f"CALL: [{arguments.name}]")
    gen(
        data_path=arguments.input,
        save_dir=arguments.output,
        limit=arguments.limit,
    )


if __name__ == "__main__":
    main()
