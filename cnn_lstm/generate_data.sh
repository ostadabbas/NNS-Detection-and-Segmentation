#rm -rf ./data/annotation
#mkdir ./data/annotation
#rm -rf ./data/image_data
#mkdir ./data/image_data

python3 utils/frameExtraction.py --fold testFold
python3 utils/frameExtraction.py --fold trainFold

python3 utils/n_frames.py --fold testFold
python3 utils/n_frames.py --fold trainFold

python3 utils/gen_anns_list.py --fold testFold
python3 utils/gen_anns_list.py --fold trainFold

python3 utils/jsonGen.py --fold testFold
python3 utils/jsonGen.py --fold testFold