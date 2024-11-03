CUDA_VISIBLE_DEVICES=0, nohup python -u main_train.py --image_dir data/iu_xray/images/ --ann_path data/iu_xray/iu_annotation_promptmrg.json --dataset_name iu_xray --num_workers 0 --gen_max_len 150 --gen_min_len 100 --batch_size 8 --epochs 6 --save_dir results/promptmrg/experiment_results/custom_iu_model --seed 456789 --init_lr 5e-5 --min_lr 5e-6 --warmup_lr 5e-7 --weight_decay 0.05 --warmup_steps 2000 --cls_weight 4 --clip_k 21 --beam_size 3 > log_promptmrg.out 2>&1    