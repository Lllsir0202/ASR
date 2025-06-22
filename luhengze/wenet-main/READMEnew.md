# We can use bash commend as following to train

``` bash
python3 ./wenet/bin/train.py --config=./20210601_u2++_conformer_libtorch_aishell/train.yaml --model_dir=./model --train_data=../SeniorTalk/sentence_data/data_train.list --cv_data=../SeniorTalk/sentence_data/data_val.list
```