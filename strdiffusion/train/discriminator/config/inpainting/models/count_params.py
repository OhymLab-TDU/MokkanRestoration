#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch

# 1) 确保可以导入 models 包
#   如果脚本与 __init__.py 同级，并且你在此目录执行 python count_params.py
#   那么不一定需要 sys.path.append。若放别处，可用下行代码:
sys.path.append("/home/shiono/StrDiffusion/train/discriminator/config/inpainting/models")

from models import create_model

def main():
    # 2) 构造 opt 字典。以下键值是根据 denoising_model.py、base_model.py 等处的使用补齐
    opt = {
        # 指定要用的模型 (在 models/__init__.py 里判断)
        "model": "denoising",

        # 常见基础配置
        "gpu_ids": None,   # None 表示用 CPU, 或 [0] 表示用单卡 GPU
        "is_train": False, # 不进行训练，只做初始化
        "dist": False,     # 不用分布式

        # 训练相关配置(即使不训练，也要把里面用到的键填上)
        "train": {
            "is_weighted": False,
            "loss_type": "MSE",
            "weight": 1.0,
            "optimizer": "Adam",    # 或 "AdamW", "Lion" 等
            "lr_G": 1e-4,
            "weight_decay_G": 0.0,
            "beta1": 0.9,
            "beta2": 0.99,

            # 学习率调度相关
            "lr_scheme": "MultiStepLR",  # 或 "TrueCosineAnnealingLR"
            "lr_steps": [10000, 20000],
            "restarts": [],
            "restart_weights": [],
            "lr_gamma": 0.5,
            "clear_state": False,
            "niter": 50000,    # 若 "TrueCosineAnnealingLR" 用到
            "eta_min": 1e-7    # 同上
        },

        # 定义生成器网络(在 define_G(opt) 里会被调用: opt["network_G"])
        "network_G": {
            # 要实例化的模型类(字符串)，在 DenoisingUNet_arch.py 中
            "which_model_G": "ConditionalUNet",
            # 传给该模型构造函数的参数字典
            "setting": {
                "in_nc": 3,
                "out_nc": 3,
                "nf": 64,
                "depth": 4,
                "upscale": 1
            }
        },

        # 路径相关配置(在 DenoisingModel.load() 等函数调用)
        "path": {
            "pretrain_model_G": None,  # 若你有预训练模型，改成 "/path/to/pretrained_G.pth"
            "strict_load": False
            # 如果项目中还有用到其他 path 键，比如 "pretrain_model_D"、"resume_state" 等，
            # 也要在这里补齐即可。
        }
    }

    # 3) 用 factory 方法创建封装后的 DenoisingModel 对象
    model = create_model(opt)

    # 4) 统计参数数量
    #    注意 DenoisingModel 本身非 nn.Module，需要访问 model.model / model.dis
    g_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in model.dis.parameters() if p.requires_grad)

    print(f"Generator trainable parameters: {g_params}")
    print(f"Discriminator trainable parameters: {d_params}")
    print(f"Total trainable parameters: {g_params + d_params}")

if __name__ == "__main__":
    main()
