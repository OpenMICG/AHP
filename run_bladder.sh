export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

result_dir='./output/bladder/'
date=$(date '+%Y%m%d%H%M%S')

python -m torch.distributed.run --nproc_per_node=4 --master_addr="localhost" --master_port=29503 report_generation.py \
       --output_dir=${result_dir}${date} \
       --image_dir="./data/bladder/images" \
       --ann_path="./data/bladder/annotation.json" \
       --dataset_name="bladder" \
       --max_seq_length=60 \
       --threshold=3 \
       --batch_size=16 \
       --seed=46 \
