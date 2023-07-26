#！/bin/bash
# nohup bash experiments_c.sh > experiments_c.log 2>&1 &

# main
echo "---Main Experiment---"
python ./HolisticRCA/trainer/c_rca_trainer.py --model_path /workspace/project/working/2023/HolisticRCA/model/c/checkpoint/main.pt --epochs 250 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 128 --efi_in_dim 128 --efi_te_heads 2 --efi_te_layers 1 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1
python ./HolisticRCA/trainer/c_localizer.py --model_path /workspace/project/working/2023/HolisticRCA/model/c/checkpoint/main.pt --epochs 250 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 128 --efi_in_dim 128 --efi_te_heads 2 --efi_te_layers 1 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1

# Ablation Study
echo "--Ablation Study 1---"
python ./HolisticRCA/ablation/c_ablation1_trainer.py
echo "--Ablation Study 2---"
python ./HolisticRCA/ablation/c_ablation2_trainer.py
echo "--Ablation Study 3---"
python ./HolisticRCA/ablation/c_ablation3_trainer.py
