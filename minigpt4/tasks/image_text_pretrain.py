"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from minigpt4.common.registry import registry
from minigpt4.tasks.base_task import BaseTask
import evaluate
import numpy as np
from minigpt4.common.logger import MetricLogger, SmoothedValue
import torch.distributed as dist
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.datasets.data_utils import prepare_sample

class Metrics_Huggingface():
    def __init__ (self, model_id = "gpt2"):
        self.isbuild = False
    
    def build_scorer(self):
        self.rouge = evaluate.load('rouge', keep_in_memory=True)
        self.bleu = evaluate.load("bleu", keep_in_memory=True)
        self.sacrebleu = evaluate.load("sacrebleu", keep_in_memory=True)
        self.bertscore = evaluate.load("bertscore", keep_in_memory=True)
        self.isbuild = True

    def compute (self, labels, preds):
        if not self.isbuild:
            self.build_scorer()
            
        scores = {}

        results = self.rouge.compute(predictions=preds, references=labels)
        for k in results.keys():
            scores[k] = round (results[k], 4)
        
        results = self.bleu.compute(predictions=preds, references=labels)
        scores['bleu'] = round (results['bleu'], 4)

        results = self.sacrebleu.compute(predictions=preds, references=labels)
        scores['sacrebleu'] = round (results['score'], 4)

        results = self.bertscore.compute(predictions=preds, references=labels, lang="en")
        scores['bertscore_p'] = round (np.mean(results ['precision']), 4)
        scores['bertscore_r'] = round (np.mean(results ['recall']), 4)
        scores['bertscore_f1'] = round (np.mean(results ['f1']), 4)
        
        return scores

@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.scorer = Metrics_Huggingface()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        print("Start evaluation at ImageTextPretrainTask.")
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        print_freq = 10

        pred = []
        gt = []

        # print(model.device, get_rank(), get_world_size(), force=True) # check evaluation device

        i = 0
        log = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            gt.extend(samples["text_answer"])
            eval_output = model.generate(images=samples["image"], texts=samples["instruction_input"])
            pred.extend(eval_output)
            print(samples["text_answer"], eval_output)
            for k in range(len(samples["instruction_input"])):
                question = samples["instruction_input"][k]
                groundtruth = samples["text_answer"][k]
                prediction = eval_output[k]
                image_path = samples["image_path"][k]
                log.append({
                    "instruction": question,
                    "GT": groundtruth,
                    "pred": prediction,
                    "image_path": image_path,
                })
            i += 1
            # save log into disk
            if i%20==0 and is_main_process():
                import json
                with open("test_result.json", "w") as f:
                    json.dump(log, f)
            results = self.scorer.compute(gt, pred)
            # if i > 0:
            #     break
            
        results["agg_metrics"] = 2.5 * results['rougeL'] + 1.5 * results['bleu']
        print(results)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

if __name__ == "__main__":
    Metrics = Metrics_Huggingface()
    print("eval start")
    predictions = [
        "They are: 2 tablespoon of parsley, 3 tablespoon of butter, 12 teaspoon of salt, 14 teaspoon of spices, 2 tablespoon of capers, 3 cup of water, 1 cup of rice, 1 tablespoon of lemoce.",
    ]
    references = [
        ["They are: 1 cup of rice, 3 cup of water, 3 tablespoon of butter, 12 teaspoon of salt, 14 teaspoon of spices, 1 tablespoon of lemoce, 2 tablespoon of capers, 2 tablespoon of parsley."],
    ]
    results = Metrics.compute(references, predictions)
    log = []
    log.append({
                    "instruction": "ins",
                    "GT": "GT",
                    "pred": "pred",
                    "image_path": "image_path",
                })
    import json
    with open("test_result_tmp.json", "w") as f:
        json.dump(log, f)
    print(results)