import sys
sys.path.append('/workspace/project/working/2023/HolisticRCA/code')
import torch

from HolisticRCA.base.base_rca_trainer import BaseRCATrainer
from HolisticRCA.explain.explainer import Explainer
from HolisticRCA.util.data_handler import copy_batch_data, rearrange_y
from shared_util.evaluation_metrics import *
import argparse
from shared_util.file_handler import FileHandler
from shared_util.seed import *
import copy


class CRCATrainer(BaseRCATrainer):
    def __init__(self, param_dict):
        super().__init__(param_dict)


if __name__ == '__main__':
    seed_everything()
    torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser()

    window_size = 6
    data_base_path = '/workspace/project/working/2023/HolisticRCA'
    parser.add_argument("--dataset_path", default=f'{data_base_path}/temp_data/2023_Eadro_SN/dataset/merge/window_size_{window_size}.pkl', type=str)
    model_base_path = FileHandler.set_folder(f'{data_base_path}/model/c')
    parser.add_argument("--model_path", default=f'{FileHandler.set_folder(model_base_path + "/checkpoint")}/main.pt', type=str)
    parser.add_argument("--window_size", default=window_size, type=int)

    parser.add_argument("--gpu", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--service_accuracy_th", default=0.2, type=float)

    parser.add_argument("--orl_te_heads", default=4, type=int)
    parser.add_argument("--orl_te_layers", default=2, type=int)
    parser.add_argument("--orl_te_in_channels", default=128, type=int)

    parser.add_argument("--efi_in_dim", default=128, type=int)
    parser.add_argument("--efi_te_heads", default=2, type=int)
    parser.add_argument("--efi_te_layers", default=1, type=int)
    parser.add_argument("--efi_out_dim", default=64 * 4, type=int)

    parser.add_argument("--eff_in_dim", default=64 * 4, type=int)
    parser.add_argument("--eff_GAT_out_channels", default=128, type=int)
    parser.add_argument("--eff_GAT_heads", default=2, type=int)
    parser.add_argument("--eff_GAT_dropout", default=0.1, type=float)

    parser.add_argument("--ec_fault_types", default=3, type=int)

    params = vars(parser.parse_args())

    rca_data_trainer = CRCATrainer(params)
    rca_data_trainer.train()
    rca_data_trainer.evaluate_rca_d3()
    ...
