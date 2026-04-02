# LLM Practice

## Dependencies

If you're running on Hipergator, first run
```bash
module load cuda/12.9.1
```
You'll have to run this command *any* time you use this codebase (not just during the initial package installation).


Install the [`uv`](https://docs.astral.sh/uv/#installation) package manager if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Use it to create a virtual environment:
```bash
uv venv
uv sync
```
This will create a local `.venv/` directory.
Run `source .venv/bin/activate` to activate the environment and `deactivate` to deactivate it.
If you're using an editor like VSCode, the path to your python interpreter is `./venv/bin/python`.

## HF Model Installs

1. Make a HuggingFace account if you don't have one already
2. Create a HuggingFace API token with read access for public repos
3. For models that require an agreement, sign it (see below sections)
4. `source ./venv/bin/activate`
5. `hf download <model name> --token <your access token>`

We discourage using mechanisms like `hf auth login` to download models because that stores your token in plain text on the machine.

### Llama
For LLama 3, go their [HuggingFace page](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and request access for their models.
For LLama 4, go their [HuggingFace page](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) and request access for their models.
