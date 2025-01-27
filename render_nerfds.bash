dataset_name='as_novel_view'
python render.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/CGC/NeRF-DS/${dataset_name}_new_corr/ --eval --iteration 30000 --skip_train --points "(122, 93)" "(50, 98)" "(86, 99)" "(143, 126)" --thetas "0.7" "0.7" "0.7" "0.7"

dataset_name='basin_novel_view'
python render.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/CGC/NeRF-DS/${dataset_name}_new_corr/ --eval --iteration 30000 --skip_train --points "(155, 46)" "(184, 63)" "(168, 77)" "(138, 67)" --thetas "0.7" "0.7" "0.7" "0.7"

dataset_name='cup_novel_view'
python render.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/CGC/NeRF-DS/${dataset_name}_new_corr/ --eval --iteration 30000 --skip_train --points "(67, 76)" --thetas "0.7"

dataset_name='plate_novel_view'
python render.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/CGC/NeRF-DS/${dataset_name}_new_corr/ --eval --iteration 30000 --skip_train --points "(67, 55)" "(122, 42)" "(151, 67)" "(131, 69)" "(117, 96)" --thetas "0.7" "0.7" "0.7" "0.7" "0.7"

dataset_name='press_novel_view'
python render.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/CGC/NeRF-DS/${dataset_name}_new_corr/ --eval --iteration 30000 --skip_train --points "(83, 49)" --thetas "0.7"
