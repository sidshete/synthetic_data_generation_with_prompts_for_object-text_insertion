#  Synthetic Scene Text/Object generation

This project automatically places synthetic **text** or **image objects** into real-world scenes based on natural language prompts such as:

> "Write the tiny bold text on truck"  
> "Place a small donut beside cake"

It uses YOLOv8, CLIP, SAM, MiDaS, and other vision/NLP models to analyze the input scene, detect reference objects, and accurately render the requested elements.

---
### To insert object
<table align="center" border="0" cellspacing="0" cellpadding="5">
<tr>
  <!-- Original Scene -->
  <td align="center">
    <img src="/results/object_placement/data_object_carrot_jpg_20250803_180809/scene.jpg" width="250"><br>
    Original Scene
  </td>

  <!-- Plus -->
  <td align="center" style="font-size: 40px; font-weight: bold; vertical-align: middle;">+</td>

  <!-- Object -->
  <td align="center">
    <img src="/synthetic_data_generation/data/object/carrot.jpg" width="250"><br>
    Object to be added
  </td>

  <!-- Plus -->
  <td align="center" style="font-size: 40px; font-weight: bold; vertical-align: middle;">+</td>

  <!-- Prompt Text + Caption -->
  <td align="center" style="font-size: 18px; vertical-align: middle;">
    <b>Prompt: </b>"Place a huge carrot left of cup"<br>
    
  </td>

  <!-- Equals -->
  <td align="center" style="font-size: 40px; font-weight: bold; vertical-align: middle;">=</td>

  <!-- Generated Image -->
  <td align="center">
    <img src="/results/object_placement/data_object_carrot_jpg_20250803_180809/composed.jpg" width="250"><br>
    Generated Image
  </td>
</tr>
</table>

<!-- Evaluation Image -->
<p align="center">
  <img src="/results/object_placement/data_object_carrot_jpg_20250803_180809/composed_yolo_debug.jpg" width="350"><br>
  Evaluation
</p>

### To insert text
<table align="center" border="0" cellspacing="0" cellpadding="5">
<tr>
  <!-- Original Scene -->
  <td align="center">
    <img src="/results/text_placement/happy_birthday_20250803_170906/scene.jpg" width="250"><br>
    Original Scene
  </td>

  <!-- Plus -->
  <td align="center" style="font-size: 40px; font-weight: bold; vertical-align: middle;">+</td>

  <!-- Object -->
  <td align="center">
    <td align="center" style="font-size: 18px; vertical-align: middle;">
    <b>Happy Birthday </b>
  </td>

  <!-- Plus -->
  <td align="center" style="font-size: 40px; font-weight: bold; vertical-align: middle;">+</td>

  <!-- Prompt Text + Caption -->
  <td align="center" style="font-size: 18px; vertical-align: middle;">
    <b>Prompt: </b>"Write the tiny bold text on cake"<br>
    
  </td>

  <!-- Equals -->
  <td align="center" style="font-size: 40px; font-weight: bold; vertical-align: middle;">=</td>

  <!-- Generated Image -->
  <td align="center">
    <img src="/results/text_placement/happy_birthday_20250803_170906/composed.jpg" width="250"><br>
    Generated Image
  </td>
</tr>
</table>

<!-- Evaluation Image -->
<p align="center">
  <img src="/results/text_placement/happy_birthday_20250803_170906/composed_ocr_debug.jpg" width="350"><br>
  Evaluation
</p>





##  Project Directory Structure

```
.
├── CLI.txt                                 # Additional python CLI
├── Dockerfile                              
├── README.md                              # Follow steps for execution and reproducibility
├── requirements.txt                        # python dependencies
├── results                                 # Already generated
│   ├── object_placement
│   └── text_placement
└── synthetic_data_generation               # main dir
    ├── data                                # dataset
    │   ├── object
    │   ├── source_images
    ├── evaluation
    │   ├── evaluate_output.py              # model evaluation
    │   ├── __init__.py
    ├── __init__.py
    ├── main.py                             # main file
    ├── models                              # pretrained weights
    │   ├── __init__.py
    │   └── sam_vit_h_4b8939.pth
    ├── outputs                             # generated results 
    ├── run_all.sh                          #--------- Batch test case execution--------
    ├── scripts
    │   ├── __init__.py
    │   ├── prompt_parser.py                # prompt analyzer
    │   ├── scene_detection.py              # detection, segmentation, depth est
    │   └── vit_test.py                     # vit object detection test file
    └── utils
        ├── blend_utils.py                  # realism techniques
        ├── image_utils.py                  # object transformation
        ├── __init__.py
        ├── midas_utils.py                  # depth estimation utils
        ├── placement_utils.py              # placement analyzer
        └── text_utils.py                   # text transformation
```

# Prerequisites
Python 3.12
Cuda 12.8
torch 12.8 + cuda
pip / virtualenv
CUDA-compatible GPU (recommended for performance)

## Dataset
The algorithm works on COCO 2017 dataset
More info and data can be downloaded from https://cocodataset.org/#home

## Dependencies
Install all Python dependencies (Docker already isntalls all the dpendencies):
```
pip install -r requirements.txt
```
If you're using the system in Docker, refer to the provided Dockerfile for containerized setup.


## Running with Docker
1. From the main fodler where Dockerfile exists run(dont forget . in the command):
```
docker build -t synthetic_data_generation .
```
2. Once it successfully builds run:
```
docker run --gpus all -it --rm -v $(pwd):/app synthetic_data_generation
```
3. Create the models folder if it does not exist:
```
mkdir -p models
```
Download the file using wget or a browser:
```
wget -O models/sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
Verify that the file `sam_vit_h_4b8939.pth` is inside the models folder before running the application.

5. In the synthetic_data_generation folder you will find `run_all.sh` which will execute batch test cases(also download and palce the model in models folder if not done manaully)
```
cd synthetic_data_generation
./run_all.sh
```
5. The results will be generated and placed in outputs folder

The system automatically runs an evaluation suite that includes:
    OCR matching (for text accuracy)
    YOLOv8 detection (object localization)
    CLIP similarity (semantic grounding)
    LPIPS (perceptual realism)
Evaluation results will be printed in the terminal and stored in logs.

6. Additional CLI can be found in python_CLI.txt which can be executed from the synthetic_data_generation where main.py file exists
eg: For object
```
cd synthetic_data_generation
python3 main.py --scene data/source_images/000000002592.jpg --object data/object/carrot.jpg --prompt "Place a huge carrot left of cup" --output outputs --evaluate
```

eg: for text
```
cd synthetic_data_generation
python3 main.py --scene data/source_images/000000065485.jpg --object "PAJERO" --prompt "Write the tiny bold text on truck" --output outputs --evaluate
```


## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.


