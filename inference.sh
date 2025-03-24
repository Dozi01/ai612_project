DATA_DIR='augmented'
DATA_SPLIT='test'
DATA_NUM=100
MODEL_NAME="gpt-4o-mini"
TEMPERATURE=0.6
SEED=1234

python -m inference --data_dir $DATA_DIR \
    --data_split $DATA_SPLIT \
    --data_num $DATA_NUM \
    --model_name $MODEL_NAME \
    --temperature $TEMPERATURE \
    --seed $SEED


# TEMPERATURE=0.2
# python -m inference --data_dir $DATA_DIR \
#     --data_split $DATA_SPLIT \
#     --model_name $MODEL_NAME \
#     --temperature $TEMPERATURE \
