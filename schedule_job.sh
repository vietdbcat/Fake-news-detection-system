#!/bin/zsh

# Kích hoạt môi trường Conda
echo "Activating environment..."
# Chạy script Python
echo "Update data..."
/home/huy31/miniconda3/envs/fakenew/bin/python /home/huy31/Projects/KDLKP/Fake-news-detection-system/utils/schedule_update_data.py

echo "Re-train data..."
/home/huy31/miniconda3/envs/fakenew/bin/python /home/huy31/Projects/KDLKP/Fake-news-detection-system/utils/train.py

# Optional: Deactivate conda environment if needed
# echo "Deactivating conda environment..."
# conda deactivate
