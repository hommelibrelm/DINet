# CUDA_VISIBLE_DEVICES=1 python tools/train.py  configs/my_experiment/oln_box.py
#all
# CUDA_VISIBLE_DEVICES=0 python tools/train.py  configs/oln_box/oln_box_transformer.py 
#convembed
# CUDA_VISIBLE_DEVICES=2 python tools/train.py  configs/oln_box/oln_box_transformer.py 

# CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh configs/my_experiment/swin_s.py 2

# tensorboard --logdir . --host 0.0.0.0 --port 8000

# python tools/train.py configs/uvo/swin_s_carafe_focal_giou_iouhead_coco_384_roihead.py

# python tools/train.py  configs/oln_box/oln_box_transformer_swin.py 

# python tools/train.py  configs/oln_box/class_agn_faster_rcnn.py

# CUDA_VISIBLE_DEVICES=1 python tools/train.py  configs/my_experiment/oln_box_cascade_uvo.py
# CUDA_VISIBLE_DEVICES=1 python tools/train.py  configs/my_experiment/swin_s_decoupled.py

# CUDA_VISIBLE_DEVICES=1,2  tools/dist_train.sh configs/my_experiment/swin_s_decoupled.py 2 

# CUDA_VISIBLE_DEVICES=1,2  tools/dist_train.sh configs/my_experiment/oln_box 2 
#1
# CUDA_VISIBLE_DEVICES=1  python tools/train.py configs/my_experiment/cascade_rpn.py
#2
# CUDA_VISIBLE_DEVICES=3  python tools/train.py configs/my_experiment/oln_box_cascade_uvo_improved_swin.py
# CUDA_VISIBLE_DEVICES=0  python tools/train.py configs/my_experiment/oln_box_cascade_uvo.py
# CUDA_VISIBLE_DEVICES=1,3 bash tools/dist_train.sh configs/my_experiment/oln_box_cascade_uvo_improved.py 2

# CUDA_VISIBLE_DEVICES=3  python tools/train.py configs/my_experiment/oln_box_uvo_improved.py

CUDA_VISIBLE_DEVICES=3  python tools/train.py configs/my_experiment/oln_box_cascade_decoupled_myroi_swin.py