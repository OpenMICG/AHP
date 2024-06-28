export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

result_dir='./output/mimic_cxr/'
date=$(date '+%Y%m%d%H%M%S')

python -m torch.distributed.run --nproc_per_node=4 --master_addr="localhost" --master_port=29503 report_generation.py \
       --output_dir=${result_dir}${date} \
       --mage_dir="./data/mimic_cxr/images" \
       --image_dir="./data/mimic_cxr/annotation.json" \
       --dataset_name="mimic_cxr" \
       --max_seq_length=100 \
       --threshold=10 \
       --batch_size=16 \
       --seed=46 \