import os
import time
import json
import argparse
from tqdm import tqdm
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from openai import OpenAI
from pyfiglet import figlet_format
import nltk
from nltk.translate.meteor_score import meteor_score


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate HERMES results')
    parser.add_argument('dir', help='Directory of the results')
    parser.add_argument('--eval_gpt', default=False, action='store_true')
    parser.add_argument('--api_key', default="xxx", help='API key for OpenAI')
    parser.add_argument("--max_files", default=None, help="Number of files to evaluate for GPT")
    parser.add_argument("--eval_future_desc", default=False, action='store_true')
    return parser.parse_args()


def print_hermes_ascii():
    print(figlet_format("HERMES", font="slant"))


class GPTEvaluation:
    """
    Class for evaluating using GPT (OpenAI API).
    """

    def __init__(self, api_key):
        self.client = OpenAI(base_url='https://api.openai-proxy.org/v1', api_key=api_key)
        self.prompts = (
            "Rate my answer based on the correct answer out of 100, with higher scores indicating that the answer is closer to the correct answer, accurate to single digits like 62, 78, etc. Output the number only.\n"
        )

    def call_chatgpt(self, chatgpt_messages, max_tokens=40, model="gpt-3.5-turbo"):
        response = self.client.chat.completions.create(
            model=model, messages=chatgpt_messages, temperature=0.6, max_tokens=max_tokens
        )
        reply = response.choices[0].message.content
        return reply

    def prepare_chatgpt_message(self, prompt):
        system_message = "an evaluator who rates my answer based on the correct answer"
        messages = [{"role": "system", "content": system_message}]
        messages.append({"role": "user", "content": prompt})
        return messages

    def forward(self, data):
        answer, GT = data
        prompt = self.prompts + f"This is the correct answer: {GT}\nThis is my answer: {answer}\n"
        messages = self.prepare_chatgpt_message(prompt)
        reply = self.call_chatgpt(messages, max_tokens=60)
        return reply.strip()


class Scorer:
    """
    Class for computing main NLP metrics for text evaluation.
    """

    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

    def compute_scores(self):
        results = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(self.gt, self.ref)
            if isinstance(method, list):
                for sc, m in zip(score, method):
                    results[m] = sc
            else:
                results[method] = score
        return results


class LanguageEvaluator:
    """
    Main evaluator for HERMES. Outputs main metrics, including Chamfer Distance and text metrics.
    Only scenarios where all 0/1/2/3s future ground truths exist are considered valid for Chamfer Distance calculation.
    (Only scenarios where all future ground truths at 0, 1, 2, and 3 seconds exist are considered for Chamfer Distance evaluation.)
    """

    def __init__(self, up_limit, eval_gpt, api_key=None):
        self.up_limit = int(up_limit) if up_limit is not None else None
        self.eval_gpt = eval_gpt
        self.api_key = api_key

    def compute_scores(self, pred_path, key, future=False):
        gts, preds = [], []
        chamfer = {
            "0": [],
            "1": [],
            "2": [],
            "3": [],
        }
        valid_count = 0
        file_list = [file for file in os.listdir(pred_path) if
                     file.endswith('.json') and not file.startswith('metrics_planning')]
        if self.up_limit:
            file_list = file_list[:self.up_limit]

        for file in tqdm(file_list, desc="Loading results"):
            with open(os.path.join(pred_path, file), 'r') as f:
                data = json.load(f)
                # Text metrics
                if key in data:
                    descs = data[key]
                    if len(descs) > 2:
                        for pair in descs[:-2]:  # Ignore the last two template pairs for fair evaluation
                            preds.append(pair['answer'])
                            gts.append(pair['gt_answer'])
                # Chamfer: Only count when all 0,1,2,3s keys exist
                if not data.get('ignore_flag', False):
                    chamfer_vals = []
                    present = True
                    for k in ["chamfer_dis_frame_0", "chamfer_dis_frame_2", "chamfer_dis_frame_4",
                              "chamfer_dis_frame_6"]:
                        if k in data:
                            chamfer_vals.append(data[k])
                        else:
                            present = False
                            break
                    if present:
                        chamfer["0"].append(chamfer_vals[0])
                        chamfer["1"].append(chamfer_vals[1])
                        chamfer["2"].append(chamfer_vals[2])
                        chamfer["3"].append(chamfer_vals[3])
                        valid_count += 1

        # Chamfer table output
        print("\n========== Chamfer Distance Results ==========")
        print(f"{'Time(s)':<8} | {'Avg Chamfer':>12} | {'Valid count':>11}")
        print("-" * 38)
        for t in ["0", "1", "2", "3"]:
            vals = chamfer[t]
            avg = np.mean(vals) if vals else None
            cnt = len(vals)
            if avg is not None:
                print(f"{t:<8} | {avg:12.4f} | {cnt:11}")
            else:
                print(f"{t:<8} | {'No Data':>12} | {cnt:11}")
        print("=" * 38)
        print("Only scenarios where all future ground truths at 0-3 seconds exist are evaluated.")
        print(f"Total valid scenarios: {valid_count}")
        print("Please note that the text metrics are evaluated on all scenarios." + "\n")

        # Text metrics table output
        print("Evaluating text metrics, please wait ~3 minutes...")
        start = time.time()
        assert len(preds) == len(gts), "Number of predictions and ground truths should be the same"
        all_preds = {i: [preds[i]] for i in range(len(preds))}
        all_gts = {i: [gts[i]] for i in range(len(gts))}
        scorer = Scorer(all_preds, all_gts)
        scores = scorer.compute_scores()
        meteor_scores = [meteor_score([nltk.word_tokenize(g)], nltk.word_tokenize(p)) for p, g in zip(preds, gts)]
        meteor_avg = np.mean(meteor_scores)
        elapsed = time.time() - start

        # Output as table, with four decimal places
        print("========== Text Evaluation Metrics ==========")
        print(f"{'Metric':<12} | {'Score':>10}")
        print("-" * 27)
        print(f"{'ROUGE_L':<12} | {scores['ROUGE_L']:10.4f}")
        print(f"{'CIDEr':<12} | {scores['CIDEr']:10.4f}")
        print(f"{'METEOR':<12} | {meteor_avg:10.4f}")
        print("=" * 27)
        print(f"\nTotal samples evaluated: {len(preds)}")
        print(f"Time used: {elapsed:.1f}s")
        if self.eval_gpt:
            print("\nEvaluating GPT score, this may take longer. Please ensure you have enough credits!")
            time.sleep(5)
            eval_gpt = GPTEvaluation(self.api_key)
            gpt_scores = []
            for data in tqdm(zip(preds, gts), total=len(preds), desc="GPT"):
                try:
                    gpt_score = float(eval_gpt.forward(data))
                    gpt_scores.append(gpt_score)
                except Exception as e:
                    print(f"GPT Evaluation error: {e}")
            if gpt_scores:
                print(f"Average GPT score: {np.mean(gpt_scores):.4f}")
        else:
            print("GPT score evaluation not enabled.")


if __name__ == '__main__':
    print_hermes_ascii()
    args = parse_args()
    pred_path = args.dir
    assert os.path.exists(pred_path)
    evaluator = LanguageEvaluator(args.max_files, args.eval_gpt, args.api_key)
    evaluator.compute_scores(pred_path, key='desc')
    if args.eval_future_desc:
        print("\nEvaluating future description...\n")
        evaluator.compute_scores(pred_path, key='desc_future', future=True)
