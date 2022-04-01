# 安装依赖
- accelerate
- chinesebert
- transformers

# 运行
```bash
accelerate launch tnews.py \
    --model_name_or_path junnyu/ChineseBERT-base \
    --max_length 256 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --seed 42 \
    --output_dir tnews_outputs \
    --num_train_epochs 3
```

# 日志
```bash
04/01/2022 14:29:25 - INFO - __main__ - Distributed environment: NO
Num processes: 1
Process index: 0
Local process index: 0
Device: cuda
Use FP16 precision: False

04/01/2022 14:29:25 - WARNING - datasets.builder - Reusing dataset clue (/root/.cache/huggingface/datasets/clue/tnews/1.0.0/03db76bb0b9390b3bb683b4d0a19625e372f58a9c4ef46a6bb32e30a76e289bd)
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 735.84it/s]
loading configuration file https://huggingface.co/junnyu/ChineseBERT-base/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/e57e90770ab4d60958d8fffe8702de812f2e1843b5885396671533d66c7799b7.0c870bc91635935fa5465f239a5efe77212df4525ee46ed5c5d8c1c3ee2a5b16
Model config ChineseBertConfig {
  "architectures": [
    "ChineseBertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "finetuning_task": "tnews",
  "glyph_embedding_dim": 1728,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1",
    "2": "LABEL_2",
    "3": "LABEL_3",
    "4": "LABEL_4",
    "5": "LABEL_5",
    "6": "LABEL_6",
    "7": "LABEL_7",
    "8": "LABEL_8",
    "9": "LABEL_9",
    "10": "LABEL_10",
    "11": "LABEL_11",
    "12": "LABEL_12",
    "13": "LABEL_13",
    "14": "LABEL_14"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_10": 10,
    "LABEL_11": 11,
    "LABEL_12": 12,
    "LABEL_13": 13,
    "LABEL_14": 14,
    "LABEL_2": 2,
    "LABEL_3": 3,
    "LABEL_4": 4,
    "LABEL_5": 5,
    "LABEL_6": 6,
    "LABEL_7": 7,
    "LABEL_8": 8,
    "LABEL_9": 9
  },
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "pinyin_map_len": 32,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.17.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 23236
}

loading file https://huggingface.co/junnyu/ChineseBERT-base/resolve/main/vocab.txt from cache at /root/.cache/huggingface/transformers/d1abde434652cc1bc400a79848d616cede97afa3218487fbd2aaf614848fa3f9.b61c41f80aae95fc6f54adffa6b9fa39fc9d75e78222b149c6225cb804f7d4d6
loading file https://huggingface.co/junnyu/ChineseBERT-base/resolve/main/tokenizer.json from cache at None
loading file https://huggingface.co/junnyu/ChineseBERT-base/resolve/main/added_tokens.json from cache at None
loading file https://huggingface.co/junnyu/ChineseBERT-base/resolve/main/special_tokens_map.json from cache at None
loading file https://huggingface.co/junnyu/ChineseBERT-base/resolve/main/tokenizer_config.json from cache at /root/.cache/huggingface/transformers/12c2ab88b7c0032126e8fc50efa745440adb5092599c93ba5b6f4625cfa96963.2d723761a6e68b8eeda7c8f299bb97e5b85fb737c78d9364509a6013e751c55e
loading configuration file https://huggingface.co/junnyu/ChineseBERT-base/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/e57e90770ab4d60958d8fffe8702de812f2e1843b5885396671533d66c7799b7.0c870bc91635935fa5465f239a5efe77212df4525ee46ed5c5d8c1c3ee2a5b16
Model config BertConfig {
  "_name_or_path": "junnyu/ChineseBERT-base",
  "architectures": [
    "ChineseBertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "glyph_embedding_dim": 1728,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "pinyin_map_len": 32,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.17.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 23236
}

loading configuration file https://huggingface.co/junnyu/ChineseBERT-base/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/e57e90770ab4d60958d8fffe8702de812f2e1843b5885396671533d66c7799b7.0c870bc91635935fa5465f239a5efe77212df4525ee46ed5c5d8c1c3ee2a5b16
Model config BertConfig {
  "_name_or_path": "junnyu/ChineseBERT-base",
  "architectures": [
    "ChineseBertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "glyph_embedding_dim": 1728,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "pinyin_map_len": 32,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.17.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 23236
}

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'BertTokenizer'. 
The class this function is called from is 'ChineseBertTokenizerFast'.
loading weights file https://huggingface.co/junnyu/ChineseBERT-base/resolve/main/pytorch_model.bin from cache at /root/.cache/huggingface/transformers/26f270dc60d51964ecc42bf19f27c50ee97ba2a99dfedd4f7ab05ab9c068f868.b005bb6a94378fa10ad5b4f1b306f8d5667e6649abb73c9e1357b5323d7d0631
Some weights of the model checkpoint at junnyu/ChineseBERT-base were not used when initializing ChineseBertForSequenceClassification: ['cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.decoder.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.bias', 'cls.predictions.decoder.bias']
- This IS expected if you are initializing ChineseBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing ChineseBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of ChineseBertForSequenceClassification were not initialized from the model checkpoint at junnyu/ChineseBERT-base and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Running tokenizer on dataset: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:05<00:00,  1.77ba/s]
Running tokenizer on dataset: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 54/54 [00:30<00:00,  1.77ba/s]
Running tokenizer on dataset: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:05<00:00,  1.73ba/s]
04/01/2022 14:30:30 - INFO - __main__ - Sample 41905 of the training set: {'input_ids': [101, 3209, 3215, 812, 1469, 3249, 6858, 782, 1762, 4495, 3833, 677, 4638, 1277, 1166, 3221, 784, 720, 8043, 3209, 3215, 812, 5543, 6084, 1057, 3249, 6858, 782, 4638, 4495, 3833, 1408, 8043, 102], 'pinyin_ids': [0, 0, 0, 0, 0, 0, 0, 0, 18, 14, 19, 12, 2, 0, 0, 0, 29, 14, 19, 12, 1, 0, 0, 0, 18, 10, 19, 5, 0, 0, 0, 0, 13, 10, 2, 0, 0, 0, 0, 0, 21, 26, 3, 0, 0, 0, 0, 0, 25, 20, 19, 12, 1, 0, 0, 0, 23, 10, 19, 2, 0, 0, 0, 0, 31, 6, 14, 4, 0, 0, 0, 0, 24, 13, 10, 19, 12, 1, 0, 0, 13, 26, 20, 2, 0, 0, 0, 0, 24, 13, 6, 19, 12, 4, 0, 0, 9, 10, 5, 0, 0, 0, 0, 0, 22, 26, 1, 0, 0, 0, 0, 0, 7, 14, 10, 2, 0, 0, 0, 0, 24, 13, 14, 4, 0, 0, 0, 0, 24, 13, 10, 19, 2, 0, 0, 0, 18, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 14, 19, 12, 2, 0, 0, 0, 29, 14, 19, 12, 1, 0, 0, 0, 18, 10, 19, 5, 0, 0, 0, 0, 19, 10, 19, 12, 2, 0, 0, 0, 23, 20, 19, 12, 2, 0, 0, 0, 23, 26, 4, 0, 0, 0, 0, 0, 21, 26, 3, 0, 0, 0, 0, 0, 25, 20, 19, 12, 1, 0, 0, 0, 23, 10, 19, 2, 0, 0, 0, 0, 9, 10, 5, 0, 0, 0, 0, 0, 24, 13, 10, 19, 12, 1, 0, 0, 13, 26, 20, 2, 0, 0, 0, 0, 18, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'labels': 2}.
04/01/2022 14:30:30 - INFO - __main__ - Sample 7296 of the training set: {'input_ids': [101, 7370, 749, 7556, 705, 8024, 3313, 3341, 2792, 3300, 4638, 2571, 6853, 1062, 1385, 6963, 2199, 948, 7308, 8043, 102], 'pinyin_ids': [0, 0, 0, 0, 0, 0, 0, 0, 8, 13, 26, 2, 0, 0, 0, 0, 17, 10, 5, 0, 0, 0, 0, 0, 24, 13, 26, 19, 4, 0, 0, 0, 11, 10, 19, 12, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 10, 14, 4, 0, 0, 0, 0, 17, 6, 14, 2, 0, 0, 0, 0, 24, 26, 20, 3, 0, 0, 0, 0, 30, 20, 26, 3, 0, 0, 0, 0, 9, 10, 5, 0, 0, 0, 0, 0, 16, 26, 6, 14, 4, 0, 0, 0, 9, 14, 4, 0, 0, 0, 0, 0, 12, 20, 19, 12, 1, 0, 0, 0, 24, 14, 1, 0, 0, 0, 0, 0, 9, 20, 26, 1, 0, 0, 0, 0, 15, 14, 6, 19, 12, 1, 0, 0, 9, 6, 20, 3, 0, 0, 0, 0, 7, 14, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'labels': 4}.
04/01/2022 14:30:30 - INFO - __main__ - Sample 1639 of the training set: {'input_ids': [101, 1313, 2428, 2600, 4415, 5811, 6832, 711, 784, 720, 2218, 7313, 6452, 704, 3681, 3428, 22591, 6482, 6569, 22973, 915, 5384, 3172, 8043, 102], 'pinyin_ids': [0, 0, 0, 0, 0, 0, 0, 0, 30, 14, 19, 4, 0, 0, 0, 0, 9, 26, 4, 0, 0, 0, 0, 0, 31, 20, 19, 12, 3, 0, 0, 0, 17, 14, 3, 0, 0, 0, 0, 0, 18, 20, 4, 0, 0, 0, 0, 0, 9, 14, 2, 0, 0, 0, 0, 0, 28, 10, 14, 4, 0, 0, 0, 0, 24, 13, 10, 19, 2, 0, 0, 0, 18, 10, 5, 0, 0, 0, 0, 0, 15, 14, 26, 4, 0, 0, 0, 0, 15, 14, 6, 19, 4, 0, 0, 0, 9, 14, 10, 2, 0, 0, 0, 0, 31, 13, 20, 19, 12, 4, 0, 0, 9, 26, 2, 0, 0, 0, 0, 0, 6, 19, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 14, 6, 19, 3, 0, 0, 0, 31, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 2, 0, 0, 0, 0, 0, 0, 17, 26, 20, 2, 0, 0, 0, 0, 24, 14, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'labels': 11}.
/usr/local/lib/python3.9/dist-packages/transformers/optimization.py:306: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
04/01/2022 14:30:36 - INFO - __main__ - ***** Running training *****
04/01/2022 14:30:36 - INFO - __main__ -   Num examples = 53360
04/01/2022 14:30:36 - INFO - __main__ -   Num Epochs = 3
04/01/2022 14:30:36 - INFO - __main__ -   Instantaneous batch size per device = 32
04/01/2022 14:30:36 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 32
04/01/2022 14:30:36 - INFO - __main__ -   Gradient Accumulation steps = 1
04/01/2022 14:30:36 - INFO - __main__ -   Total optimization steps = 5004
04/01/2022 14:32:37 - INFO - __main__ - epoch 0: {'accuracy': 0.5687}
Configuration saved in tnews_outputs/config.json
Model weights saved in tnews_outputs/pytorch_model.bin
tokenizer config file saved in tnews_outputs/tokenizer_config.json
Special tokens file saved in tnews_outputs/special_tokens_map.json
04/01/2022 14:34:40 - INFO - __main__ - epoch 1: {'accuracy': 0.5744}
Configuration saved in tnews_outputs/config.json
Model weights saved in tnews_outputs/pytorch_model.bin
tokenizer config file saved in tnews_outputs/tokenizer_config.json
Special tokens file saved in tnews_outputs/special_tokens_map.json
04/01/2022 14:36:43 - INFO - __main__ - epoch 2: {'accuracy': 0.5742}
Configuration saved in tnews_outputs/config.json
Model weights saved in tnews_outputs/pytorch_model.bin
tokenizer config file saved in tnews_outputs/tokenizer_config.json
Special tokens file saved in tnews_outputs/special_tokens_map.json
```

