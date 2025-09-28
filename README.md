# Symbolic_rave
Poligon for master thesis on symbolic gen music model

## Configuration

`poetry self update` - update poetry
`poetry env use $(which python)` - set python version
`poetry init` - create new project
`poetry install` - install dependencies
`poetry shell` - activate virtual environment
`poetry update` - update dependencies

## Development

`poe test` - run tests
`poe ruff` - format code
`poe mypy` - check types
`poe collect-dataset` - collect dataset

## Usage

```sh
export PYTHONPATH="/home/kuba/repo/music/Symbolic_rave:$PYTHONPATH"
```
## Knowledge

- Elbo explained: [link](docs/elbo_docs.md)