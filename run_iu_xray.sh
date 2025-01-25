export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

result_dir='./output/iu_xray/'
date=$(date '+%Y%m%d%H%M%S')

python -m torch.distributed.run --nproc_per_node=4 --master_addr="localhost" --master_port=29503 report_generation.py \
       --output_dir=${result_dir}${date} \
       --image_dir="./data/iu_xray/images" \
       --ann_path="./data/iu_xray/annotation.json" \
       --dataset_name="iu_xray" \
       --max_seq_length=60 \
       --threshold=3 \
       --batch_size=8 \
       --seed=46 \
