import os

import pandas as pd
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from tqdm import tqdm
from transformers import AutoTokenizer
import soundfile as sf

from pkg.args import args
from pkg.device import get_device
from huggingface_hub import login




def gen(
        data_path: str,
        save_dir: str,
        limit: int,
):
    login(token="xxx")
    device = get_device()
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts-pretrained").to(device)
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts-pretrained")


    df = pd.read_csv(data_path)

    marking_list = [{"id": "", "text": "", "wav_path": ""} for _ in range(len(df))]

    if limit is None:
        limit = len(df)
    limit = min(limit, len(df))
    pbar = tqdm(range(limit))
    for idx in pbar:
        row = df.iloc[idx]
        id, text = row

        postfix = f"telugu_indic_tts/{idx}.wav"

        wav_dir = os.path.join(save_dir, "telugu_indic_tts")
        os.makedirs(wav_dir, exist_ok=True)
        path_name = os.path.join(wav_dir, f"{idx}.wav")


        prompt_input_ids = tokenizer(text, return_tensors="pt").to(device)

        generation = model.generate(
            prompt_input_ids=prompt_input_ids.input_ids,
            prompt_attention_mask=prompt_input_ids.attention_mask,
        )

        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(path_name, audio_arr, model.config.sampling_rate)

        marking_list[idx] = {
            "id": id,
            "text": text,
            # not absolute path !
            "wav_path": postfix
        }



    marking = pd.DataFrame(marking_list)
    marking.to_csv(os.path.join(save_dir, "telugu_indic_tts.csv"), index=False)



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


