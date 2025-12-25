# ezmsg-lsl

Interface to Lab Streaming Layer [LSL](https://labstreaminglayer.readthedocs.io/) in [ezmsg](https://github.com/iscoe/ezmsg).

## Installation

```bash
pip install ezmsg-lsl
```

Or install the latest development version:

```bash
pip install git+https://github.com/ezmsg-org/ezmsg-lsl@dev
```

### Dependencies
* `ezmsg`
* `pylsl`
* `numpy`

## Usage

See the `examples` folder for more details.

## Developers

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for development. It is not strictly required, but if you intend to contribute to ezmsg-lsl then using `uv` will lead to the smoothest collaboration.

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
2. Fork ezmsg-lsl and clone your fork to your local computer.
3. Open a terminal and `cd` to the cloned folder.
4. `uv sync` to create a .venv and install dependencies.
5. (Optional) Install pre-commit hooks: `uv run pre-commit install`
6. After editing code and making commits, Run the test suite before making a PR: `uv run pytest tests`
   * Currently, there are no substantial tests. 
