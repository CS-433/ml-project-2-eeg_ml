# Gender-classification...
You can find the codes here for gender classification from EEG data. We implement 4 models, including a linear model (LIN), a multilayer perceptron (MLP), a GRU and a CNN. Before running the code, you need to download the dataset from [Data1](https://drive.google.com/file/d/1zQi72b9_j1zbEUPtQorYEv29_3OLVOe6/view?usp=sharing) and [Data2](https://drive.google.com/file/d/1Y5UCXA82ko64fAdmeH0Kn4C2-EN-tVJJ/view?usp=sharing) and put them under *./data/training_data*.

## Prerequisites
- Python 3
- panda
- openpyxl
- numpy
- collection
- matplotlib 2.1.1
- scipy 1.7.1
- PyTorch 1.7.1 

Other versions may also work.

## Getting Started
### Training
To train the classifier, run
```
python  main.py --classifier CLASSIFIER_NAME --train_mode full(or window/channel) --split_num SPLIT_NUMBER --save_model 
```

The default optimizer is Adam optimizer, the learning rate is 0.001, the batch size is 128, the number of epoch is 50 (but you can change them by setting command-line arguments). If you do not want to save the trained model, drop **--save_model** (the test accuracy will be printed on the screen anyway). Otherwise, the trained model will be saved at *./checkpoints* automatically and be named in the format as *CNN_split_0_best.pth*. For more information of options, check *./lib/options.py*.

### Testing
To test the classifier, run
```
python  eval.py --classifier CLASSIFIER_NAME --train_mode full(or window/channel) --load_path PATH_TO_CHECKPOINTS 
```

The script will load the model trained on different folds from *PATH_TO_CHECKPOINTS* to evaluate on the test set. The test accuracy will be shown on the screen.

### Visualization
```
python generate_guided_backprop.py --load_path checkpoints/CNN/random --gif 
```
<p align="center">
  <img src="figs/heat-sign.png">
</p>


<p align="center">
  <img height="200" src="figs/topographic.gif">
</p>

### Statistic Analysis
```
python statistic_analysis.py --analysis ttest --fig_path ./figs
```


<!---
## Getting Started
### Installing
Clone this repo:

```bash
git clone ...
cd ReenactGAN
```

### Training
The bounday encoder is trained on WFLW and Helen dataset, and both of the boundary transformer and decoder are trained on [CelebV Dataset](https://drive.google.com/file/d/1jQ6d76T5GQuvQH4dq8_Wq1T0cxvN0_xp/view?usp=sharing). The training of the encoder requires a huge amount of time, so you can get the pretrained encoder at *./pretrained_models/v8_net_boundary_detection.pth*. 

To train the boundary transformer, run
```bash
sh script/train_Transformer.sh
```
You need to take care of the arguments **--root_dir** and **--which_target**.  **--root_dir** refers to the directory of the dataset, and **--which_target** refers to which person to be the target
```bash
0: Emmanuel_Macron
1: Kathleen
2: Jack_Ma
3: Theresa_May
4: Donald_Trump
```

To train the decoder, run
```bash
sh script/train_Decoder.sh
```
Also, you need to take care of the value of **--root_dir**, which refers to the directory of the target person.

### Testing
To test the model, run
```bash
sh script/move_models.sh ./checkpoints/Transformer_2019-xx-xx_xx-xx-xx/G_BA_xx.pth ./checkpoints/Decoder_2019-xx-xx_xx-xx-xx/xx_net_G.pth trump
sh script/test.sh
```
The images used for testing is at ./test_imgs/samples/image, and the text file, ./test_imgs/samples/images_list.txt, contains the list of these images. After the testing, you will get a floder named **results**, which contains the images of the real and reenacted faces, the boundaries and the transformed boundaries of the real faces. Here are some results.

<img src='imgs/results.png' width="1000px">

You can get our trained models from [Decoder](https://drive.google.com/file/d/1MBWABJK9webZxAMvN9Cl5FBhXateppzu/view?usp=sharing) and [Transformer](https://drive.google.com/open?id=1v-8kh0N56alKiSoBAENXp9KNJ0lg_Qtq).

-->
