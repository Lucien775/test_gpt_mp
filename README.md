# test_gpt_mp

The goal is to compare several precision and architecture variants of a GPT model, using a trained checkpoint and a text dataset.
To do this, we build several models, load them from the same checkpoint, then measure their prediction error on text batches.

This project relies in particular on the [mptorch](https://github.com/mptorch/mptorch) library.

## Folder Structure

- **main.py**
  - CLI entry point
  - parses arguments, loads the checkpoint, selects the experiment and launches the analysis
- **builder.py**
  - builds the models associated with each experiment
  - centralises instantiation and weight-loading logic
- **error_analyzer.py**
  - loads the text data
  - generates batches
  - runs the models and compares their losses
- **formats.py**
  - contains helpers for creating quantisation formats and LayerNorm
- **model_up.py**
  - reference model with uniform precision
- **model_mp_block.py**
  - model with mixed precision per block (attention / FFN / LN)
- **model_mp_mhsa.py**
  - model with mixed precision at the multi-head attention level
- **model_mp_ln.py**
  - model with LayerNorm in mixed precision
- **model_similarity_mp.py**
  - model with similarity in mixed precision
- **model_mlp_mp.py**
  - model with MLPs in mixed precision

## How It Works

1. The `main.py` script loads a checkpoint as a dictionary containing:
   - the model configuration
   - the model state
   - the stoi/itos mappings
2. It selects an experiment via the `--experiment` argument.
3. The associated builder creates a collection of models, each with its own specific configuration.
4. `ErrorAnalyzer` runs these models on batches and measures the average loss.
5. Results are saved to a CSV file.

## Running

From this folder:

```bash
python main.py --experiment uniform
python main.py --experiment mixed_block
python main.py --experiment mhsa
python main.py --experiment ln
```

Useful arguments:
- `--model_path`: path to the `.pt` checkpoint
- `--data_path`: path to the text file
- `--batch_size`: batch size
- `--num_batches`: number of batches evaluated
- `--output`: path to the output CSV
- `--no-cuda`: force execution on CPU

Example:
```bash
python main.py --experiment ln --batch_size 32 --num_batches 100 --output test_results/ln_custom.csv
```

## Adding a New Experiment

To add a new model or experiment, follow this procedure:

1. **Create a new model file**
   - e.g. `model_mp_xxx.py`
   - define a dataclass for the configuration
   - implement the model blocks and the forward logic

2. **Add a corresponding builder in `builder.py`**
   - create a function of the form `build_models_xxx(cfg, checkpoint_state, device)`
   - instantiate the model with the desired formats
   - return a `name -> model` dictionary

3. **Wire the experiment into `main.py`**
   - add the builder import
   - add the entry to `EXPERIMENTS`
   - add the corresponding value to the `choices` of `--experiment`

4. **Optionally set an output file**
   - define a CSV name in `EXPERIMENTS` to keep the results

5. **Test the experiment**
   - run `python main.py --experiment <name>`