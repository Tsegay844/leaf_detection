python3 espdet_run.py \
  --class_name leaf \
  --pretrained_path None \
  --dataset "datasets/grape_leaf/data.yaml" \
  --size 320 320 \
  --target "esp32s3" \
  --calib_data "deploy/grape_leaf_calib" \
  --espdl "espdet_pico_320_320_grape_leaf.espdl" \
  --img "espdet.jpg"