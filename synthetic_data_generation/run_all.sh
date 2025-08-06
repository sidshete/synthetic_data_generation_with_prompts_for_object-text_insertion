#!/bin/bash

MODEL_DIR="./models"
MODEL_FILE="sam_vit_h_4b8939.pth"
MODEL_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Check if models directory exists, if not create it
if [ ! -d "$MODEL_DIR" ]; then
  echo "Creating models directory..."
  mkdir -p "$MODEL_DIR"
fi

# Check if model file exists, if not download it
if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
  echo "Model file not found. Downloading..."
  wget -O "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
else
  echo "Model file already exists."
fi

python3 main.py --scene data/source_images/000000065485.jpg --object "PAJERO" --prompt "Write the tiny bold text on truck" --output outputs --evaluate

python3 main.py --scene data/source_images/000000064718.jpg --object "YONEX" --prompt "Write the tiny text diagonal on person" --output outputs --evaluate

python3 main.py --scene data/source_images/000000064718.jpg --object "Yonex" --prompt "Write the tiny text rotate 90 on person" --output outputs --evaluate

python3 main.py --scene data/source_images/000000046031.jpg --object "RESARO" --prompt "Write the tiny text tilted below laptop" --output outputs --evaluate

python3 main.py --scene data/source_images/000000046031.jpg --object "RESARO" --prompt "Write the tiny red text tilted above mouse" --output outputs --evaluate

python3 main.py --scene data/source_images/000000002157.jpg --object "Happy Birthday" --prompt "Write the tiny bold text on cake" --output outputs --evaluate

python3 main.py --scene data/source_images/000000000139.jpg --object "Vase" --prompt "Write the tiny text rotate 90 on vase" --output outputs --evaluate

python3 main.py --scene data/source_images/000000000139.jpg --object "Wall of fame" --prompt "Write tiny blue text on top of tv" --output outputs --evaluate

python3 main.py --scene data/source_images/000000000139.jpg --object "Wooden floor" --prompt "Write tiny text below tv" --output outputs --evaluate

python3 main.py --scene data/source_images/000000002157.jpg --object data/object/Donut.jpg --prompt "Place the small donut beside cake" --output outputs --evaluate

python3 main.py --scene data/source_images/000000022969.jpg --object data/object/bear3.jpg --prompt "Place the large bear below giraffe" --output outputs --evaluate

python3 main.py --scene data/source_images/000000022969.jpg --object data/object/bear3.jpg --prompt "Place the large bear beside giraffe" --output outputs --evaluate

python3 main.py --scene data/source_images/000000016451.jpg --object data/object/surfboard.jpg --prompt "Place a large surfboard above backpack" --output outputs --evaluate

python3 main.py --scene data/source_images/000000002592.jpg --object data/object/carrot.jpg --prompt "Place a huge carrot left of cup" --output outputs --evaluate

python3 main.py --scene data/source_images/000000027186.jpg --object data/object/cat.jpg --prompt "Place a large cat beside person" --output outputs --evaluate

python3 main.py --scene data/source_images/000000027186.jpg --object data/object/cat.jpg --prompt "Place a medium cat left of person" --output outputs --evaluate

