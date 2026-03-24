# DeepLenseSim

Software to simulate strong lensing images for the DeepLense project.

## Data sets

Scripts used to create data sets and link to data sets are located in each of the Model_# folders.

## Installation

First install [colossus](https://bdiemer.bitbucket.io/colossus/cosmology_cosmology.html), [lenstronomy](https://github.com/sibirrer/lenstronomy), and [pyHalo](https://github.com/dangilman/pyHalo)

```console
foo@bar:~$ pip install colossus
foo@bar:~$ pip install lenstronomy==1.9.2
foo@bar:~$ git clone https://github.com/dangilman/pyHalo.git
foo@bar:~$ cd pyHalo
foo@bar:~/pyHalo$ python setup.py develop
foo@bar:~/pyHalo$ cd ..
```


Then clone this repository to your machine and install with setup.py

```console
foo@bar:~$ git clone https://github.com/mwt5345/DeepLenseSim.git
foo@bar:~$ cd DeepLenseSim
foo@bar:~/DeepLenseSim$ python setup.py install
```

## Papers
[![](https://img.shields.io/badge/arXiv-1909.07346%20-red.svg)](https://arxiv.org/abs/1909.07346) [![](https://img.shields.io/badge/arXiv-2008.12731%20-red.svg)](https://arxiv.org/abs/2008.12731) [![](https://img.shields.io/badge/arXiv-2112.12121%20-red.svg)](https://arxiv.org/abs/2112.12121)

## Agentic Workflow

This repository now includes a `deeplense_agent` package that wraps DeepLenseSim with a natural-language Pydantic AI workflow.

### Strategy

The agent uses a two-step orchestration flow:

1. It interprets a free-form user request into a typed `SimulationRequest`.
2. It calls `preview_simulation_plan(...)` to resolve defaults and summarize the exact run, then waits for human confirmation before calling `run_deeplense_simulation(...)`.

The execution layer writes:

- One `.npy` array per generated image
- One `.png` preview per generated image
- One `run_metadata.json` file with structured metadata
- One `contact_sheet.png` summarizing the run

### Supported Model Configurations

The agent currently supports:

- `Model_I`
- `Model_II`
- `Model_III`

`Model_IV` is intentionally not exposed through the agent because it depends on the external Galaxy10 DECals dataset, which is not bundled in this repository.

### Environment Variables

The agent prefers Gemini by default and falls back to Ollama:

- `GOOGLE_API_KEY` or `GEMINI_API_KEY` for Gemini
- `OLLAMA_BASE_URL` for Ollama, defaulting to `http://localhost:11434/v1`
- `OLLAMA_API_KEY` if your Ollama endpoint expects one

Default model stack:

- Gemini: `gemini-2.5-flash`
- Ollama fallback: `qwen3:8b`

### Running The Agent

Install the optional agent dependencies:

```console
pip install -e .[agent]
```

Start an interactive session:

```console
deeplense-agent
```

For a specific provider
```console
deeplense-agent --provider ollama
```

Or if we want to run the directly:
```console
python -m deeplense_agent --provider ollama --ollama-model qwen3:8b
```
Or provide a one-shot prompt:

```console
deeplense-agent "Generate 2 Euclid-like CDM lensing images at 64 pixels with lens redshift 0.5 and source redshift 1.2"
```

The agent will ask follow-up questions if the request is underspecified, preview the resolved plan, and only run the simulation after you confirm.
