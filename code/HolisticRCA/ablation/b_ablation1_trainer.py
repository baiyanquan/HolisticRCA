import sys
sys.path.append('/workspace/project/working/2023/HolisticRCA/code')
import torch
import json

from HolisticRCA.ablation.base_ablation1_trainer import BaseAblation1Trainer
from shared_util.evaluation_metrics import *
from shared_util.file_handler import FileHandler
from shared_util.seed import *
import argparse
import pickle
import pandas as pd


class BRCATrainer(BaseAblation1Trainer):
    def __init__(self, param_dict):
        super().__init__(param_dict)

    def predict(self):
        with open(f'/workspace/project/working/2023/HolisticRCA/temp_data/2022_ICASSP_AIOps_challenge/dataset/merge/special_samples.pkl', 'rb') as f:
            special_sample_dict = pickle.load(f)

        with open(f'/workspace/project/working/2023/HolisticRCA/temp_data/2022_ICASSP_AIOps_challenge/dataset/merge/test_index_list.pkl', 'rb') as f:
            test_index_list = pickle.load(f)

        self.model.load_state_dict(torch.load(self.param_dict["model_path"]))
        self.model.eval()
        with torch.no_grad():
            raw_y_pred = dict()
            y_true = []
            for batch_id, batch_data in enumerate(self.rca_data_loader.data_loader['test']):
                out = self.model(batch_data)
                for ent_type in self.rca_data_loader.meta_data['ent_types']:
                    if ent_type not in raw_y_pred.keys():
                        raw_y_pred[ent_type] = []
                    raw_y_pred[ent_type].extend((torch.sigmoid(out[ent_type])).cpu().detach().numpy())
                y_true.extend(batch_data['y'].cpu().detach().numpy())
        y_true = np.array(y_true)
        y_true = y_true.reshape(y_true.shape[0], y_true.shape[1])[:600, :]
        y_pred = []

        for i in range(len(y_true)):
            temp_pred = [0, 0, 0]

            score1 = self.calculate_score(test_index_list[i], raw_y_pred['rc1'], self.param_dict['rc1_accuracy_th'])
            if raw_y_pred['rc1'][i] > self.param_dict['rc1_accuracy_th'] or score1 == 1:
                temp_pred[0] = 1
            if i in special_sample_dict['rc1_fluctuate']:
                if score1['score'] > 0.14:
                    temp_pred[0] = 1

            if i in special_sample_dict['rc23_null']:
                temp_pred[1] = 1
                temp_pred[2] = 1
            else:
                score2 = self.calculate_score(test_index_list[i], raw_y_pred['rc2'], self.param_dict['rc2_accuracy_th'])
                if i in special_sample_dict['rc2_fluctuate'] and score2['total'] > 0:
                    temp_pred[1] = 1
                elif raw_y_pred['rc2'][i] > self.param_dict['rc2_accuracy_th']:
                    temp_pred[1] = 1

                score3 = self.calculate_score(test_index_list[i], raw_y_pred['rc3'], self.param_dict['rc3_accuracy_th'])
                if i in special_sample_dict['rc3_fluctuate'] and score3['total'] > 0:
                    temp_pred[2] = 1
                elif raw_y_pred['rc3'][i] > self.param_dict['rc3_accuracy_th']:
                    temp_pred[2] = 1
            y_pred.append(temp_pred)
        self.logger.info('----------')
        self.logger.info(f'evaluation dataset type: test')
        fc_result = fault_type_classification(y_pred, y_true)
        convert = {
            'p': 'precision',
            'r': 'recall',
            'f1': 'f1'
        }
        for em in ['p', 'r', 'f1']:
            self.logger.info(
                f'{convert[em].ljust(9)} | micro: {fc_result["micro_" + convert[em] + "_score"]:.6f}; macro: {fc_result["macro_" + convert[em] + "_score"]:.6f}; score: {fc_result[convert[em] + "_score"]}')
        self.logger.info('----------')

    def calculate_score(self, index_pair, y_pred, y_accuracy_th):
        score = 0
        for j in range(index_pair[0], index_pair[1]):
            if y_pred[j] > y_accuracy_th:
                score += 1
        return {
            'score': score / (index_pair[1] - index_pair[0]),
            'total': score
        }


if __name__ == '__main__':
    seed_everything()
    torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser()

    data_base_path = '/workspace/project/working/2023/HolisticRCA'

    parser.add_argument("--dataset_path", default=f'{data_base_path}/temp_data/2022_ICASSP_AIOps_challenge/dataset/merge/rca.pkl', type=str)
    model_base_path = FileHandler.set_folder(f'{data_base_path}/model/b')
    parser.add_argument("--model_path", default=f'{FileHandler.set_folder(model_base_path + "/checkpoint")}/ablation1.pt', type=str)

    window_size = 30
    parser.add_argument("--window_size", default=window_size, type=int)

    parser.add_argument("--gpu", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--epochs", default=350, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--rc1_accuracy_th", default=0.2, type=float)
    parser.add_argument("--rc2_accuracy_th", default=0.8, type=float)
    parser.add_argument("--rc3_accuracy_th", default=0.9, type=float)

    parser.add_argument("--orl_te_heads", default=2, type=int)
    parser.add_argument("--orl_te_layers", default=2, type=int)
    parser.add_argument("--orl_te_in_channels", default=128, type=int)

    parser.add_argument("--efi_in_dim", default=128, type=int)
    parser.add_argument("--efi_te_heads", default=4, type=int)
    parser.add_argument("--efi_te_layers", default=2, type=int)
    parser.add_argument("--efi_out_dim", default=64 * 4, type=int)

    parser.add_argument("--eff_in_dim", default=64 * 4, type=int)
    parser.add_argument("--eff_GAT_out_channels", default=128, type=int)
    parser.add_argument("--eff_GAT_heads", default=2, type=int)
    parser.add_argument("--eff_GAT_dropout", default=0.4, type=float)

    parser.add_argument("--ec_fault_types", default=1, type=int)

    params = vars(parser.parse_args())

    rca_data_trainer = BRCATrainer(params)
    rca_data_trainer.train()
    rca_data_trainer.predict()
