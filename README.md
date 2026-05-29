# ViViT & Video Swin on SSv2

Fine-tuning ViViT-B and VideoMAE-B (Video Swin proxy) on Something-Something V2 (SSv2) dataset. Compares Full Fine-Tuning vs LoRA with efficiency metrics.

## Files
- `vivit_ssv2_final.ipynb` — ViViT-B colab sub-test (5 epochs, 40k clips)
- `videoswin_colab_metrics.ipynb` — VideoMAE-B colab sub-test (5 epochs, 40k clips)
- `vivit_fullfinetuned_ssv2.py` — ViViT Full FT script for RTX 5090 (35 epochs)
- `vivit_lora_ssv2.py` — ViViT LoRA script for RTX 5090 (35 epochs)

## Results (Colab Sub-Test)
| Model | Top-1 | Top-5 | FT Time |
|-------|-------|-------|---------|
| ViViT-B (head-only) | 14.97% | 37.48% | ~8h |
| VideoMAE-B (Full FT) | 43.51% | 73.66% | 12.26h |

## References
- Arnab et al., ViViT, ICCV 2021
- Liu et al., Video Swin Transformer, CVPR 2022
- Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, ICLR 2022. https://arxiv.org/abs/2106.09685
