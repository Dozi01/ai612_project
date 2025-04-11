DATA_DIR='augmented'
DATA_SPLIT='test'

DATA_NUM=100
DATA_NULL_RATIO=0.45 # 45% of the data is null

MODEL_NAME="gpt-4o-mini"
TEMPERATURE=0.6

THRESHOLD_FOR_CLASSIFICATION=30
IS_HARD_CLASSIFICATION=False # True: hard classification(answerable/unanswerable), False: soft classification(score)

MAX_RETRY=2
NUM_CONSISTENCY_CHECK=3
SEED=1004

#### RETRIEVER
TOP_K=10
HYBRID_WEIGHT=0.5

python -m inference --data_dir $DATA_DIR \
    --data_split $DATA_SPLIT \
    --data_num $DATA_NUM \
    --data_null_ratio $DATA_NULL_RATIO \
    --model_name $MODEL_NAME \
    --temperature $TEMPERATURE \
    --threshold_for_classification $THRESHOLD_FOR_CLASSIFICATION \
    --is_hard_classification $IS_HARD_CLASSIFICATION \
    --max_retry $MAX_RETRY \
    --num_consistency_check $NUM_CONSISTENCY_CHECK \
    --seed $SEED \
    --retriever_top_k $TOP_K \
    --hybrid_weight $HYBRID_WEIGHT \
