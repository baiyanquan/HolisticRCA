#！/bin/bash
# nohup bash experiments_b.sh > experiments_b.log 2>&1 &

# main
echo "---Main Experiment---"
python ./HolisticRCA/trainer/b_rca_trainer.py --model_path /workspace/project/working/2023/HolisticRCA/model/b/checkpoint/main.pt --epochs 350 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 128 --efi_in_dim 128 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.4

# Ablation Study
echo "--Ablation Study 1---"
python ./HolisticRCA/ablation/b_ablation1_trainer.py
echo "--Ablation Study 2---"
python ./HolisticRCA/ablation/b_ablation2_trainer.py
echo "--Ablation Study 3---"
python ./HolisticRCA/ablation/b_ablation3_trainer.py

# Sensitivity Analysis
echo "---------------------"
echo "orl_te_in_channels: 32"
python ./HolisticRCA/trainer/b_rca_trainer.py --model_path /workspace/project/working/2023/HolisticRCA/model/b/checkpoint/orl_te_in_channels_32.pt --epochs 350 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 32 --efi_in_dim 32 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.4
echo "---------------------"
echo "orl_te_in_channels: 64"
python ./HolisticRCA/trainer/b_rca_trainer.py --model_path /workspace/project/working/2023/HolisticRCA/model/b/checkpoint/orl_te_in_channels_64.pt --epochs 350 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 64 --efi_in_dim 64 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.4
echo "---------------------"
echo "orl_te_in_channels: 256"
python ./HolisticRCA/trainer/b_rca_trainer.py --model_path /workspace/project/working/2023/HolisticRCA/model/b/checkpoint/orl_te_in_channels_256.pt --epochs 350 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.4
echo "---------------------"

echo "---------------------"
echo "eff_in_dim: 32"
python ./HolisticRCA/trainer/b_rca_trainer.py --model_path /workspace/project/working/2023/HolisticRCA/model/b/checkpoint/eff_in_dim_32.pt --epochs 350 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 128 --efi_in_dim 128 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 32 --eff_in_dim 32 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.4
echo "---------------------"e
cho "eff_in_dim: 64"
python ./HolisticRCA/trainer/b_rca_trainer.py --model_path /workspace/project/working/2023/HolisticRCA/model/b/checkpoint/eff_in_dim_64.pt --epochs 350 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 128 --efi_in_dim 128 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 64 --eff_in_dim 64 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.4
echo "---------------------"
echo "eff_in_dim: 128"
python ./HolisticRCA/trainer/b_rca_trainer.py --model_path /workspace/project/working/2023/HolisticRCA/model/b/checkpoint/eff_in_dim_128.pt --epochs 350 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 128 --efi_in_dim 128 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 128 --eff_in_dim 128 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.4
echo "---------------------"
