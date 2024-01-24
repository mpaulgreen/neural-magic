<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

---

version: 1.1.0

# General Variables
num_epochs: &num_epochs 13
init_lr: 1.5e-4 
final_lr: 0

qat_start_epoch: &qat_start_epoch 8.0
observer_epoch: &observer_epoch 12.0
quantize_embeddings: &quantize_embeddings 1

distill_hardness: &distill_hardness 1.0
distill_temperature: &distill_temperature 2.0

# Modifiers:

training_modifiers:
  - !EpochRangeModifier
      end_epoch: eval(num_epochs)
      start_epoch: 0.0

  - !LearningRateFunctionModifier
      start_epoch: 0
      end_epoch: eval(num_epochs)
      lr_func: linear
      init_lr: eval(init_lr)
      final_lr: eval(final_lr)
    
quantization_modifiers:

  - !QuantizationModifier
      start_epoch: eval(qat_start_epoch)
      disable_quantization_observer_epoch: eval(observer_epoch)
      freeze_bn_stats_epoch: eval(observer_epoch)
      quantize_embeddings: eval(quantize_embeddings)
      quantize_linear_activations: 0
      exclude_module_types: ['LayerNorm', 'Tanh']
      submodules:
        - bert.embeddings
        - bert.encoder
        - bert.pooler
        - classifier


distillation_modifiers:
  - !DistillationModifier
     hardness: eval(distill_hardness)
     temperature: eval(distill_temperature)
     distill_output_keys: [logits]

constant_modifiers:

  - !ConstantPruningModifier
      start_epoch: 0.0
      params: __ALL_PRUNABLE__

---

# 90% Pruned Quantized oBERT base uncased

This model is the result of transferring and quantizing a pruned 90% oBERT base uncased model for text classification on QQP.

This model is from [The Optimal BERT Surgeon](https://arxiv.org/abs/2203.07259) paper.

# Training command

```bash
sparseml.transformers.train.text_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --distill_teacher zoo:nlp/text_classification/obert-base/pytorch/huggingface/qqp/base-none \
  --task_name qqp \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --logging_steps 1000 \
  --save_steps 1000 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --max_seq_length 128 \
  --recipe zoo:nlp/text_classification/obert-base/pytorch/huggingface/qqp/pruned90_quant-none \
  --output_dir obert_base_pruned90_quant_qqp \
  --preprocessing_num_workers 32 \
  --seed 10194
```

## Evaluation

The model could be evaluated with the following command:

```bash
sparseml.transformers.train.text_classification \
  --output_dir dense_bert-text_classification_qqp_eval \
  --model_name_or_path zoo:nlp/text_classification/obert-base/pytorch/huggingface/qqp/pruned90_quant-none \
  --task_name qqp --max_seq_length 128 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_eval 
```

