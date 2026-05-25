from comet import download_model, load_from_checkpoint
from comet.models import CometModel
from evaluate import load
from sacrebleu.metrics import BLEU

import torch
import argparse
import numpy as np
import json
import gc
import os

import warnings

warnings.filterwarnings("ignore")

from translationese_eval import TranslationeseIndex


def comet_scorer(data, model: CometModel, model_name: str, ref_free: bool = False):
    print(f"Calculating {model_name}")
    if ref_free:
        data = list(
            map(
                lambda x: {"src": x["src"].strip(), "mt": x["mt"].strip()},
                data,
            )
        )
    else:
        if args.evaluate_refs:
            return "--"
        data = list(
            map(
                lambda x: {
                    "src": x["src"].strip(),
                    "ref": x["ref"].strip(),
                    "mt": x["mt"].strip(),
                },
                data,
            )
        )
    scores = model.predict(data)
    torch.cuda.empty_cache()
    gc.collect()
    print("Done!")
    return scores.system_score


def load_comet_model(model_path):
    model_path = download_model(model_path)
    model = load_from_checkpoint(model_path)

    return model


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--tgt_lang", type=str, default=None)
    parser.add_argument("--evaluate_refs", action="store_true", default=False)
    parser.add_argument(
        "--output_dir", type=str, help="Path to result file(s).", default=None
    )
    parser.add_argument("--output_file", type=str, default=None)
    return parser


meteor = load("meteor")
spbleu = BLEU(tokenize="spBLEU-1K")
teval = TranslationeseIndex(
    model_dir="t_index_reproduce/models/sft/qwen2.5-0.5b-mixture-5000-10",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
comet = load_comet_model("Unbabel/wmt22-comet-da")
xcomet = load_comet_model("Unbabel/XCOMET-XL")
cometkiwi = load_comet_model("Unbabel/wmt22-cometkiwi-da")

code2name = {
    "enzh": "Chinese",
    "enfr": "French",
    "ende": "German",
    "enms": "Malay",
    "enko": "Korean",
}


def calc_spbleu(data):
    if args.evaluate_refs:
        return "--"
    print("Calculating spBLEU")
    mts = [item["mt"].strip() for item in data]
    refs = [[item["ref"].strip()] for item in data]
    scores = spbleu.corpus_score(mts, refs)
    torch.cuda.empty_cache()
    gc.collect()
    print("Done!")
    return scores.score


def calc_meteor(data):
    if args.evaluate_refs:
        return "--"
    print("Calculating METEOR")

    scores = []
    for item in data:
        score = meteor.compute(
            predictions=[item["mt"].strip()], references=[item["ref"].strip()]
        )
        scores.append(score["meteor"])

    results = np.mean(scores) * 100
    
    torch.cuda.empty_cache()
    gc.collect()
    print("Done!")
    return results.item()


def calc_teval(data, tgt_lang):
    srcs = [item["src"] for item in data]
    mts = [item["mt"] for item in data]
    granularity = ["segment", "sequence"]
    results = {}
    for g in granularity:
        res = teval.score(srcs, mts, lang=tgt_lang, granularity=g)
        results.update({g: res["mean_score"] * 100})
    return results


parser = init_parser()
args = parser.parse_args()

print(torch.cuda.is_available())


def run_evaluation(data, tgt_lang):
    comet_score = comet_scorer(data, comet, "Comet")
    cometkiwi_score = comet_scorer(data, cometkiwi, "CometKiwi", ref_free=True)
    xcomet_score = comet_scorer(data, xcomet, "XCOMET", ref_free=args.evaluate_refs)
    spbleu_score = calc_spbleu(data)
    meteor_score = calc_meteor(data)
    teval_score = calc_teval(data, tgt_lang)

    final = {
        "COMET": comet_score * 100,
        "CometKiwi": cometkiwi_score * 100,
        "XCOMET": xcomet_score * 100,
        "spBLEU": spbleu_score,
        "METEOR": meteor_score,
        "Naturalness": teval_score,
    }

    print(final)

    return final


def process_jsonl(data_file):
    processed = []
    with open(data_file, "r") as file:
        data = []
        for line in file.readlines():
            data.append(json.loads(line))

    for item in data:
        processed.append(
            {
                "src": item["source"].strip(),
                "mt": item["foreignization"].strip(),
                "ref": item["domestication"].strip(),
            }
        )

    return processed


def prepare_data(data):
    ratios = []
    comet_ = []
    scaled = []
    for item in data:
        ratios.append(
            {"src": item["src"], "mt": item["mts"]["ratio"], "ref": item["ref"]}
        )
        comet_.append(
            {"src": item["src"], "mt": item["mts"]["comet"], "ref": item["ref"]}
        )
        scaled.append(
            {"src": item["src"], "mt": item["mts"]["scaled"], "ref": item["ref"]}
        )
    return ratios, comet_, scaled


if __name__ == "__main__":
    if os.path.isdir(args.data_path):
        data_dirs = os.listdir(args.data_path)
        for model_dir in data_dirs:
            model_path = f"{args.data_path}/{model_dir}"
            for data_dir in os.listdir(model_path):
                decode_path = f"{model_path}/{data_dir}"
                for data_file in os.listdir(decode_path):
                    data_path = f"{args.data_path}/{model_dir}/{data_dir}/{data_file}"
                    if data_file.endswith("jsonl"):
                        data = process_jsonl(data_path)
                    else:
                        with open(data_path, "r") as file:
                            data = json.load(file)

                        lang_idx = (
                            data_file.index("_en") + 1 if not "none" in data_path else 0
                        )
                        lang = slice(lang_idx, lang_idx + 4)
                        tgt_lang = code2name[data_file[lang]]

                    if not "none" in data_path:
                        ratios, qe, scaled = prepare_data(data)
                        results = {
                            "ratio": run_evaluation(ratios, tgt_lang),
                            "comet": run_evaluation(qe, tgt_lang),
                            "scaled": run_evaluation(scaled, tgt_lang),
                        }
                    else:
                        results = run_evaluation(data, tgt_lang)

                    if "evaluation/scores" not in args.output_dir:
                        output_path = f"evaluation/scores/{args.output_dir}"
                    else:
                        output_path = args.output_dir
                    file_name = "_".join(data_path.split("/")[2:]).replace(
                        ".json", "_scores.json"
                    )
                    result_path = f"{output_path}/{file_name}"
                    os.makedirs(output_path, exist_ok=True)
                    with open(result_path, "w") as file:
                        json.dump(results, file, indent=2)
    else:
        if args.data_path.endswith("jsonl"):
            data = process_jsonl(args.data_path)
        else:
            with open(f"{args.data_path}", "r") as file:
                data = json.load(file)

        results = run_evaluation(data)

        file_extension = args.data_path.split(".")[-1]
        if args.output_dir:
            if args.output_file:
                if not args.output_file.endswith(file_extension):
                    output_file = f"{args.output_file}.json"
                result_path = f"{args.output_dir}/{args.output_file}"
            else:
                output_file = args.data_path.split("/")[-1].replace(
                    file_extension, "_scores.json"
                )
                result_path = f"{args.output_dir}/{output_file}"
        elif args.output_file:
            output_path = "evaluation/scores"
            result_path = output_path + "/" + args.output_file
        else:
            assert ValueError(
                "`args.output_dir` and `args.output_file` cannot both be None."
            )
        os.makedirs(output_path, exist_ok=True)
        with open(result_path, "w") as file:
            json.dump(results, file, indent=2)
