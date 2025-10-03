# Synthetic Graph Demo

## Setup Instructions

Download `concept_embeddings.npy` from:
- https://drive.google.com/file/d/1DPN7UkeEDDaWbligd0eXDyabAamKXM8Q/view?usp=sharing

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the app

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open your browser and go to `http://localhost:8000`.

## Notes

- Ensure you have the `concept_embeddings.npy` file in the same directory as `main.py`.
- The maximum number of nodes for the graph is set to 20. You can adjust this in the `MAX_NODES` variable in `main.py`.
