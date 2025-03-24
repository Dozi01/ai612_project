DATA_DIR='original'
DATA_SPLIT='valid'
DATA_NUM=30
MODEL_NAME="gpt-4o"
TEMPERATURE=0.6

python -m inference --data_dir $DATA_DIR \
    --data_split $DATA_SPLIT \
    --model_name $MODEL_NAME \
    --temperature $TEMPERATURE \
