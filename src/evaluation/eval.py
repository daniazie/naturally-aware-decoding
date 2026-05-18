from comet import download_model, load_from_checkpoint
from comet.models import CometModel
from evaluate import load
from sacrebleu.metrics import BLEU
from pathlib import Path
import torch
import argparse
import numpy as np
import json
import gc
import os

from translationese_eval import TranslationeseIndex

def comet_scorer(data, model: CometModel, model_name: str, ref_free: bool = False):
    print(f"Calculating {model_name}")
    if ref_free:
        data = list(map(lambda x: {'src': x['src'].strip(), 'mt': x['mt'].strip()}, data))
    else:
        data = list(map(lambda x: {'src': x['src'].strip(), 'ref': x['ref'].strip(), 'mt': x['mt'].strip()}, data))
    scores = model.predict(data)
    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')
    return scores.system_score

def load_comet_model(model_path):
    model_path = download_model(model_path)
    model = load_from_checkpoint(model_path)

    return model

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--tgt_lang', type=str, default=None)
    parser.add_argument('--output_dir', type=str, help="Path to result file(s).", default=None)
    parser.add_argument('--output_file', type=str, default=None)
    return parser

meteor = load('meteor')
spbleu = BLEU(tokenize='spBLEU-1K')
teval = TranslationeseIndex(model_path="t_index_reproduce/models/sft/qwen2.5-0.5b-mixture-5000-10", device_map="auto")
comet = load_comet_model("Unbabel/wmt22-comet-da")
xcomet = load_comet_model("Unbabel/XCOMET-XL")
cometkiwi = load_comet_model("Unbabel/wmt22-cometkiwi-da")

code2name = {
    "enzh": "Chinese",
    "enfr": "French",
    "ende": "German"
}

def calc_spbleu(data):
    print("Calculating spBLEU")
    mts = [item['mt'].strip() for item in data]
    refs = [[item['ref'].strip()] for item in data]
    scores = spbleu.corpus_score(mts, refs)
    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')
    return scores.score

def calc_meteor(data):
    print("Calculating METEOR")

    scores = []
    for item in data:
        score = meteor.compute(predictions=[item['mt'].strip()], references=[item['ref'].strip()])
        scores.append(score['meteor'])

    results = np.mean(scores) * 100
    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')
    return results.item()

def calc_teval(data, tgt_lang):
    srcs = [item['src'] for item in data]
    mts = [item['mt'] for item in data]
    results = teval.score(srcs, mts, lang=tgt_lang, normalise_score=True)
    return results['mean_score']
    

parser = init_parser()
args = parser.parse_args()

print(torch.cuda.is_available())

def run_evaluation(data, tgt_lang):
    comet_score = comet_scorer(data, comet, "Comet")
    cometkiwi_score = comet_scorer(data, cometkiwi, "CometKiwi", ref_free=True)
    xcomet_score = comet_scorer(data, xcomet, "XCOMET")
    spbleu_score = calc_spbleu(data)
    meteor_score = calc_meteor(data)
    teval_score = calc_teval(data, tgt_lang)

    final = {
        "COMET": comet_score * 100,
        "CometKiwi": cometkiwi_score * 100,
        "XCOMET": xcomet_score * 100,
        'spBLEU': spbleu_score,
        'METEOR': meteor_score,
        "Naturalness": teval_score * 100
    }

    print(final)

    return final

def process_jsonl(data_file):
    processed = []
    with open(data_file, 'r') as file:
        data = []
        for line in file.readlines():
            data.append(json.loads(line))

    for item in data:
        processed.append({
            "src": item['source'].strip(),
            "mt": item['foreignization'].strip(),
            "ref": item['domestication'].strip()
        })

    return processed

if __name__ == "__main__":
    if os.path.isdir(args.data_path):
        data_files = os.listdir(args.data_path)
        for data_file in data_files:
            if data_file.endswith("jsonl"):
                data = process_jsonl(f'{args.data_path}/{data_file}')
            else:
                with open(f'{args.data_path}/{data_file}', 'r') as file:
                    data = json.load(file)
                
                tgt_lang = code2name[data_file.split('_')[2]]

            results = run_evaluation(data, tgt_lang)

            if not "evaluation/scores" in args.output_dir:
                output_path = f'evaluation/scores/{args.output_dir}'
            else:
                output_path = args.output_dir
            file_name = data_file.split('/')[-1].replace('.json', '_scores.json')
            result_path = f"{output_path}/{file_name}"
            os.makedirs(output_path, exist_ok=True)
            with open(result_path, 'w') as file:
                json.dump(results, file, indent=2)
    else:
        if args.data_path.endswith("jsonl"):
            data = process_jsonl(args.data_path)
        else:
            with open(f"{args.data_path}", "r") as file:
                data = json.load(file)

        results = run_evaluation(data)
        
        file_extension = args.data_path.split('.')[-1]
        if args.output_dir:
            if args.output_file:
                if not args.output_file.endswith(file_extension):
                    output_file = f'{args.output_file}.json'
                result_path = f"{args.output_dir}/{args.output_file}"
            else:
                output_file = args.data_path.split('/')[-1].replace(file_extension, '_scores.json')
                result_path = f"{args.output_dir}/{output_file}"
        elif args.output_file:
            output_path = "evaluation/scores"
            result_path = output_path + '/' + args.output_file
        else:
            assert ValueError("`args.output_dir` and `args.output_file` cannot both be None.")
        os.makedirs(output_path, exist_ok=True)
        with open(result_path, 'w') as file:
            json.dump(results, file, indent=2)