# Hand Gesture Recognition with simple CNN network
### A project for Computer Vision EE3077 class
Model trained on Google Colab's GPU <br>
Input stream taken from smartphone's IP camera <br>
<br>
Link drive contain model file : [Google Drive](https://drive.google.com/drive/folders/10rUQV73oLwi1XFerXkNZnDuQVz3XTg6f?usp=sharing) <br>
Trained using this Kaggle dataset : [leapgestrecog](https://www.kaggle.com/gti-upm/leapgestrecog)
<br>
Detail can be found in my PowerPoint : [pptx]
<br>
Foreground extraction using <b>Skin color detection </b> (HSV and YCrCb) combined with <b>MOG2 background subtractor</b>. I choose this approach instead of feeding everything into a neural network because I don't fond of using black box model which I don't understand.<br>
Skin color based detection works well against noisy background but unstable against varying lighting condition. Although I use forehead skin as a color reference, this still require meticulous tuning correct offset. (which is the purpose of function controlPanel())<br>
MOG2 background subtractor using Gaussian model to extract dynamic foreground from still background, which require a very static background or else noise will be introduced in the system. <b>
Combine both of these methods will theoretically eliminate weakness of each method, but require manual fine-tuning. 
### DOWNLOAD GUIDE
1. I advise create a virtual environment first to test this project 
2. Clone this repository or just the main.py alone is fine. 
3. Download my pretrained model in the above link. 
4. Store the model folder in the same directory/ folder as the main.py script. Or if you want to use any other model, just open the main.py script and change the *model = model_load("path/to/your/model")*
5. My notebook is there if you want to have a look at my result, how I trained and evaluated it, or retrained it yourself.
 ```bash
python3 -m venv cv202venv
cd cv202venv
source bin/activate
git clone https://github.com/Minatozaki0911/CV202.git
cd CV202
pip install requirement.txt
python3 main.py
```
### USER GUIDE
After successfully run the main.py script, you should be able to open your webcam and see a green rectangle area. That is our region of interest. Then : <br>
- press key 'b' to capture the background for background subtraction. This should contain the background only, do not put your face, hand or any non static object when capture the background
- press key 'r' to reset the background if you accidentally mess up, then recapture it by press 'b' again 
- press 'q' to exit script
