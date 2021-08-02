# Hand Gesture Recognition with a simple CNN network
### A project for Computer Vision EE3077 class
Model trained on Google Colab's GPU <br>
Input image taken from live webcam <br>
<br>
Link drive contain model file : https://drive.google.com/drive/folders/10rUQV73oLwi1XFerXkNZnDuQVz3XTg6f?usp=sharing
### USING GUIDE
1. I advise create a virtual environment first to test this project 
2. Clone this repository or just the main.py alone is fine. 
3. Download my pretrained model in the above link. 
4. Store the model folder in the same directory/ folder as the main.py script. Or if you want to use any other model, just open the main.py script and change the *model = model_load("path/to/your/model")*
5. My notebook is there if you want to have a look at my result, how I trained and evaluated it, or retrained it yourself.
 ```bash
python3 -m venv cv202
cd cv202
git clone https://github.com/Minatozaki0911/CV202.git
pip install requirement.txt
python3 main.py
```
