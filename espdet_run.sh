python3 espdet_run.py \
  --class_name leaf \
  --pretrained_path "runs/detect/train/weights/best.pt" \
  --dataset "datasets/grape_leaf/data.yaml" \
  --size 256 320 \
  --target "esp32s3" \
  --calib_data "deploy/grape_leaf_calib" \
  --espdl "espdet_pico_320_256_grape_leaf.espdl" \
  --img "espdet.jpg"