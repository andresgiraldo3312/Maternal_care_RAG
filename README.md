# Maternal care: Is it good to trust in clinical guidelines-based recommendations given by IA?

This repository contains the necessary code to reproduce the pipeline for our paper title "Maternal care: Is it good to trust in clinical guidelines-based
recommendations given by IA?"

# Pipeline

<img width="614" alt="image" src="https://github.com/user-attachments/assets/05f8bca5-9eae-4008-8730-fe70ec1a5b69" />

# Reproducibility

1. Create an environment and install the required packages.

```bash
python -m venv RAG_guias
source RAG_guias/bin/activate
pip install -r requirements.txt
```

2. Install ollama and download LLM.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma2:2b
```

3. Execute the pipeline. When running local models, ensure that the system is equipped with a GPU offering adequate VRAM, as the memory requirements will vary depending on the size and architecture of the selected LLM. For cloud-based deployments, appropriate authentication credentials must be properly configured to enable model access.

```bash
python main.py Data/KNOWLEDGED_BASE/ Data/QUESTION/preguntas.xlsx Data/ANSWER/ Data/GROUND_TRUTH/ground_truth.xlsx config.yml
```

## Citation

If you use this code in a scientific publication, we would appreciate citations to the following paper:

<pre>
@article{Pending,
  title={Maternal care: Is it good to trust in clinical guidelines-based recommendations given by IA?},
  author={ },
  journal={Revista Biomédica},
  pages={Pending},
  year={2025}
}
</pre>
