# python tools/test.py configs/oln_box/oln_box_transformer.py work_dirs/oln_box_transformer/epoch_7.pth --show-dir local_results --show-score-thr 0.7

# python tools/test.py configs/oln_box/oln_box_transformer.py work_dirs/oln_box_transformer/epoch_8.pth  --show-dir local_results

# CUDA_VISIBLE_DEVICES=0  python tools/test.py configs/my_experiment/oln_box_cascade.py work_dirs/oln_box_decoupled/epoch_4.pth --eval bbox

# CUDA_VISIBLE_DEVICES=0  python tools/test.py configs/my_experiment/oln_box.py work_dirs/oln_box/epoch_4.pth --eval bbox

CUDA_VISIBLE_DEVICES=3 python tools/test.py configs/my_experiment/oln_box_cascade_uvo_improved.py /mnt/disk/lm/mmdetection-2.20.0/work_dirs/oln_box_cascade_uvo_improved_decoupled_lr002/epoch_6.pth --show-dir local_results --show-score-thr 0.7

# CUDA_VISIBLE_DEVICES=3 python tools/test.py configs/my_experiment/oln_box.py work_dirs/oln_box/epoch_6.pth --show-dir local_results --show-score-thr 0.7