# VoD eval
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python tools/all_test_evaluate.py \
  --config configs/RCDFNet/RCDFNet_VoD.py \
  --checkpoint checkpoint/vod \
  --out work_dirs/260320_RCDFNet_BMA_vod_deformableattention_eval_out \
  --format-only \
  --eval-options \
    pklfile_prefix=work_dirs/260320_RCDFNet_BMA_vod_deformableattention_eval_out/prefix \
    submission_prefix=work_dirs/260320_RCDFNet_BMA_vod_deformableattention_eval_out/submission

# TJ4D eval
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python tools/test.py \
  --config configs/RCDFNet/RCDFNet_TJ4D.py \
  --checkpoint checkpoint/TJ4D/RCDFNet.pth \
  --eval mAP 