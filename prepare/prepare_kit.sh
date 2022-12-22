# down kit https://motion-annotation.humanoids.kit.edu/dataset/
# refer to https://github.com/anindita127/Complextext2animation

# use kit_dataset_path = /apdcephfs/share_1227775/shingxchen/datasets/KIT/KIT-ML-Default
python src/data.py -- xx

python dataProcessing/meanVariance.py -mask '[0]' -feats_kind rifke -dataset KITMocap -path2data /apdcephfs/share_1227775/shingxchen/datasets/KIT/KIT-ML-Default -f_new 8