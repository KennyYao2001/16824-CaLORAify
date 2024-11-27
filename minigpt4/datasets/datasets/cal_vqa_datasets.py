"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import csv
import random

from PIL import Image

from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class CALVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.istrain = "train" in vis_root

        self.instruction_pool =[
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

        exist_annotation = []
        for idx, ann in enumerate(self.annotation):
            for imageIdx, imgPath in enumerate(ann["image_paths"]):
                exist_annotation.append((idx, imageIdx))
        self.annotation, self.info = exist_annotation, self.annotation

        # load question pool from csv file
        self.question_pool = []
        with open(os.path.join(vis_root, f"../Cal_data_Question_Set.csv"), 'r') as f:
            csvreader = csv.reader(f)
            for i, row in enumerate(csvreader):
                if i==0:
                    continue
                question = {
                    "question_id": int(row[0]),
                    "question": row[1],
                    "question_style": row[2],
                }
                self.question_pool.append(question)
        self.num_question = len(self.question_pool)

    def get_data(self, data_index, question_index):
        sampleNum, imageNum = self.annotation[data_index]
        ann = self.info[sampleNum]

        image_path = os.path.join(self.vis_root, f"../train", ann["image_paths"][imageNum])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        question_id = question_index
        question = self.question_pool[question_id]['question']

        answers = ann["gt_ingredient_answer"]

        return {
            "image_path": image_path,
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answers,
        }


    def __getitem__(self, index):
        if self.istrain:
            data_index = index // self.num_question
            question_index = index % self.num_question
        else:
            data_index = index 
            question_index = random.randint(0, len(self.question_pool)-1)
        data = self.get_data(data_index, question_index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image_path": data['image_path'],
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
            "text_answer": data['answer'],
        }

        return data 
    
    def __len__(self):
        return len(self.annotation) * self.num_question if self.istrain else len(self.annotation)
