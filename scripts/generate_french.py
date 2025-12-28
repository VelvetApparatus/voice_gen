from __future__ import annotations

import os

import pandas as pd
from tqdm import tqdm

from pkg.args import args
from pkg.device import get_device
from kokoro import KPipeline
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

    pipeline = KPipeline(lang_code='f', device=device)

    if limit is None:
        limit = len(df)
    limit = min(limit, len(df))
    pbar = tqdm(range(limit))
    for idx in pbar:
        row = df.iloc[idx]
        id, text = row

        postfix = f"french_kokoro/{idx}.wav"

        wav_dir = os.path.join(save_dir, "french_kokoro")
        os.makedirs(wav_dir, exist_ok=True)
        path_name = os.path.join(wav_dir, f"{idx}.wav")

        generator = pipeline(text, voice='ff_siwis')
        for i, (gs, ps, audio) in enumerate(generator):
            print(i, gs, ps)
            sf.write(path_name, audio, 24000)

        marking_list[idx] = {
            "id": id,
            "text": text,
            # not absolute path !
            "wav_path": postfix
        }

    marking = pd.DataFrame(marking_list)
    marking.to_csv(os.path.join(save_dir, "french_kokoro.csv"), index=False)


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
