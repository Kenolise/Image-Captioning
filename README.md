<h1 align="center"> Image Captioning </h1> 
 
 ***An encoder-decoder based model to caption images built using PyTorch and deployed using Streamlit. ***
 
 ***This model uses inceptionV3 as encoder and LSTM layers as decoder. This model is trained on Flickr30k dataset.***

## Demo

 <p align="center"> <img src="https://github.com/Kenolise/Image-Captioning/blob/master/test_examples/pawpaw.jpg" width="500" height="320"  /> </p>
<p align="center"> <b>Prediction: <i>a boy wailing with hand on head .</i></b>
 
<p align="center"> <img src="https://github.com/Kenolise/Image-Captioning/blob/master/test_examples/pic.jpg" width="500" height="320"  /> </p>
<p align="center"> <b>Prediction: <i>a man holding a paper and rejecting a drink offer .</i></b>



## Running on native machine
### dependencies
* python3
* `python -m spacy download en` - for tokenizing english sentences  

### pip packages
`pip install -r requirements.txt` 

## Steps to train your own model
  ### Scripts
  `neuralnet/train.py` - is used to train the model  
  
  `engine.py` - is used to perform inference 
 
 `ui.py` - is used to build the streamlit app
    
  For more details make sure to visit these files to look at script arguments and description
  
  1. Dataset  
    i. Download the [Flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset) dataset  
    ii. Remove the duplicate images folder and csv file  
  
  2. Training  
    use train.py to train the model  
