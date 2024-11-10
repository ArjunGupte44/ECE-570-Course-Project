import logging
import os
from abc import abstractmethod
import numpy as np
import time

import cv2
import torch
import json
from .metrics_clinical import CheXbertMetrics

VQA_RAD_MRG_TEST_LOG = r"C:\Users\agupt\OneDrive\Documents\Purdue\Junior Year\Fall Semester\ECE 570\Course Project\ECE_570_Course_Project\results\promptmrg\experiment_results\base_iu_model\test\base_iu_model_vqa_rad_mrg_test_log.json"

class BaseTester(object):
    def __init__(self, model, criterion_cls, metric_ftns, args, device):
        self.args = args
        self.model = model
        self.device = device

        self.chexbert_metrics = CheXbertMetrics('./checkpoints/stanford/chexbert/chexbert.pth', args.batch_size, device)

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)


        self.criterion_cls = criterion_cls
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

class Tester(BaseTester):
    def __init__(self, model, criterion_cls, metric_ftns, args, device, test_dataloader):
        super(Tester, self).__init__(model, criterion_cls, metric_ftns, args, device)
        self.test_dataloader = test_dataloader

    def test_blip(self):
        self.logger.info('Start to evaluate in the test set.')
        file = open(VQA_RAD_MRG_TEST_LOG, "w")
        image_report_dict = {}
        log = dict()
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images, image_names, captions, cls_labels, clip_memory) in enumerate(self.test_dataloader):
                try:
                    images = images.to(self.device) 
                    clip_memory = clip_memory.to(self.device) 
                    ground_truths = captions
                    reports, _, _ = self.model.generate(images, clip_memory, sample=False, num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len)
                    torch.cuda.empty_cache()

                    test_res.extend(reports)
                    test_gts.extend(ground_truths)

                    self.logger.info('{}/{}'.format(batch_idx, len(self.test_dataloader)))

                    for gen_report, gt_report, image_name in zip(reports, ground_truths, image_names):
                        self.logger.info(f"\nGenerated:\n{gen_report}\nGround Truth:\n{gt_report}\n\n")
                        image_report_dict[image_name] = gen_report

                    print(f"Finished: {16 * batch_idx}/107")
                
                except Exception as e:
                    print(e)
            
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            test_ce = self.chexbert_metrics.compute(test_gts, test_res)
            
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            log.update(**{'test_' + k: v for k, v in test_ce.items()})
            self.logger.info(len(test_res))
            print(f"Finished: {len(test_res)}/107")

            #Dump image_report dict into json
            json.dump(image_report_dict, file, indent=4)
            file.close()
        return log

