# make input structure
make input ARGS="--replays sample --include-components worker ground air building vision neutral resource terrain"

# make label structure
make label ARGS="--replays sample --method all_correct"

# training Mask RCNN model (reproduce from Joo et al 2023)
# set nvidia visible device at infra/.env file
make train ARGS=" \
  --replays sample \
  --include-components worker ground air building vision \
  --label-method all_correct \
  --max-epoch 20 \
  --window-size 4 \
  --interval 8 \
  --batch-size 16 \
  --sample-ratio 1.0 \
  --log-level log \
  "
# OR
# use NVIDIA_VISIBLE_DEVICES={device number}
NVIDIA_VISIBLE_DEVICES=0 make train ARGS=" \
  --replays sample \
  --include-components worker ground air building vision \
  --label-method all_correct \
  --max-epoch 20 \
  --window-size 4 \
  --interval 8 \
  --batch-size 16 \
  --sample-ratio 1.0 \
  --log-level log \
  "

# inference from trained model
make inference ARGS=" \
  --replays sample \
  --include-components worker ground air building vision \
  --model-name {model_name} \
  --model-number 30 \
  --window-size 4 \
  --label-method all_correct \
  --batch-size 16 \
  --score-threshold 0.0 \
  --sample-ratio 1.0 \
  --output-dir /workspace/predictions \
  "
# OR
NVIDIA_VISIBLE_DEVICES=0 make inference ARGS=" \
  --replays sample \
  --include-components worker ground air building vision \
  --model-name {model_name} \
  --model-number 30 \
  --window-size 4 \
  --label-method all_correct \
  --batch-size 16 \
  --score-threshold 0.0 \
  --sample-ratio 1.0 \
  --output-dir /workspace/predictions \
  "