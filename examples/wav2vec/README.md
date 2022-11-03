# ccc-wav2vec 2.0

## Training a new model with the CLI tools

Given a directory containing wav files to be used for pretraining (we recommend splitting each file into separate file 10 to 30 seconds in length)

### Prepare training data manifest

First, install the `soundfile` library:

```shell script
pip install soundfile
```

Next, run:

```shell script
python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext $ext --valid-percent $valid
```

$ext should be set to flac, wav, or whatever format your dataset happens to use that soundfile can read.

$valid should be set to some reasonable percentage (like 0.01) of training data to use for validation.
To use a pre-defined validation set (like dev-other from librispeech), set to it 0 and then overwrite valid.tsv with a
separately pre-processed manifest file.

### Train a ccc-wav2vec 2.0 base model

Note that the input is expected to be single channel, sampled at 16 kHz

```shell script
$ fairseq-hydra-train \
    task.data=/path/to/data \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/pretraining \
    --config-name wav2vec2_base_librispeech
```

Note: you can simulate 64 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 64/k

### Train a ccc-wav2vec 2.0 large model

```shell script
$ fairseq-hydra-train \
    task.data=/path/to/data \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/pretraining \
    --config-name wav2vec2_large_librivox
```

Note: you can simulate 128 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 128/k

### Parameters of interest

* The $\alpha$, $\beta$ and $\gamma$ values from the paper can be caliberated from the `cc_weights` parameter in the `criterion` section of the pre-training configs which can be found from the [pre-training config](https://github.com/Speech-Lab-IITM/CCC-wav2vec-2.0/examples/wav2vec/config/pretraining).
* The `cluster_factor` and `scale_factor` parameters can be modified from the `model` section of the pre-training configs which can be found from the [pre-training config](https://github.com/Speech-Lab-IITM/CCC-wav2vec-2.0/examples/wav2vec/config/pretraining).
* The augmentations used for ccc-wav2vec 2.0 requires the noise set of MUSAN dataset. The path to the same is to be specified in the `path_to_musan_noise_set` variable of the __getitem__ method of the [raw_audio_dataset](https://github.com/Speech-Lab-IITM/CCC-wav2vec-2.0/fairseq/data/audio/raw_audio_dataset.py) file.

### Fine-tune a pre-trained model with CTC

Fine-tuning a model requires parallel audio and labels file, as well as a vocabulary file in fairseq format.
A letter vocabulary can be downloaded [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt).
An example [script](libri_labels.py) that generates labels for the Librispeech dataset from the tsv file produced by wav2vec_manifest.py can be used as follows:

```shell script
split=train
$ python libri_labels.py /path/to/tsv --output-dir /output/dir --output-name $split
```

Fine-tuning on 100h of Librispeech with letter targets:

```shell script
$ fairseq-hydra-train \
    distributed_training.distributed_port=$PORT \
    task.data=/path/to/data \
    model.w2v_path=/path/to/model.pt \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/finetuning \
    --config-name base_100h
```

There are other config files in the config/finetuning directory that can be used to fine-tune on other splits.
You can specify the right config via the `--config-name` parameter.

Note: you can simulate 24 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 24/k

Decoding with a language model during training requires flashlight [python bindings](https://github.com/facebookresearch/flashlight/tree/master/bindings/python) (previously called [wav2letter](https://github.com/facebookresearch/wav2letter).
If you want to use a language model, add `+criterion.wer_args='[/path/to/kenlm, /path/to/lexicon, 2, -1]'` to the command line.

### Evaluating a CTC model

Evaluating a CTC model with a language model requires [flashlight python bindings](https://github.com/facebookresearch/flashlight/tree/master/bindings/python) (previously called [wav2letter](https://github.com/facebookresearch/wav2letter) to be installed.

Fairseq transformer language model used in the wav2vec 2.0 paper can be obtained from the [wav2letter model repository](https://github.com/facebookresearch/wav2letter/tree/master/recipes/sota/2019).
Be sure to upper-case the language model vocab after downloading it.

Letter dictionary for pre-trained models can be found [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt).

Next, run the evaluation command:

```shell script
$subset=dev_other
python examples/speech_recognition/infer.py /checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw --task audio_finetuning \
--nbest 1 --path /path/to/model --gen-subset $subset --results-path /path/to/save/results/for/sclite --w2l-decoder kenlm \
--lm-model /path/to/kenlm.bin --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
--post-process letter
```

To get raw numbers, use --w2l-decoder viterbi and omit the lexicon. To use the transformer language model, use --w2l-decoder fairseqlm.
