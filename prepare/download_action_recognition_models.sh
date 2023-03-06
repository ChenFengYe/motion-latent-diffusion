# from https://github.com/Mathux/ACTOR/blob/master/prepare/download_recognition_models.sh

mkdir -p deps/actionrecognition/
cd deps/actionrecognition/

echo -e "The action recognition models will be stored in the 'models/actionrecognition
/' folder\n"

# NTU13 action recognition models
echo -e "Downloading the HumanAct12 action recognition model"
wget https://raw.githubusercontent.com/EricGuo5513/action-to-motion/master/model_file/action_recognition_model_humanact12.tar -O humanact12_gru.tar
echo -e

echo -e "Downloading the UESTC action recognition model"
gdown "https://drive.google.com/uc?id=1bSSD69s1dHY7Uk0RGbGc6p7uhUxSDSBK"
echo -e

echo -e "Downloading done!"
