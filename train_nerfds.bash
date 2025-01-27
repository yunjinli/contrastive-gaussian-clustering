dataset_name="basin_novel_view"
python train.py -s /mnt/sda/4dgsam_data/NeRF-DS/$dataset_name -m /mnt/sda/4dgsam_output/CGC/NeRF-DS/${dataset_name}_new_corr/ --eval --load_iteration 20000 --iteration 30000 --test_iterations 20100 30000 --save_iteration 20100 30000 --contrastive_interval 1

dataset_name="cup_novel_view"
python train.py -s /mnt/sda/4dgsam_data/NeRF-DS/$dataset_name -m /mnt/sda/4dgsam_output/CGC/NeRF-DS/${dataset_name}_new_corr/ --eval --load_iteration 20000 --iteration 30000 --test_iterations 20100 30000 --save_iteration 20100 30000 --contrastive_interval 1

dataset_name="plate_novel_view"
python train.py -s /mnt/sda/4dgsam_data/NeRF-DS/$dataset_name -m /mnt/sda/4dgsam_output/CGC/NeRF-DS/${dataset_name}_new_corr/ --eval --load_iteration 20000 --iteration 30000 --test_iterations 20100 30000 --save_iteration 20100 30000 --contrastive_interval 1

dataset_name="press_novel_view"
python train.py -s /mnt/sda/4dgsam_data/NeRF-DS/$dataset_name -m /mnt/sda/4dgsam_output/CGC/NeRF-DS/${dataset_name}_new_corr/ --eval --load_iteration 20000 --iteration 30000 --test_iterations 20100 30000 --save_iteration 20100 30000 --contrastive_interval 1