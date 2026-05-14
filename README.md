# Vindex — inférence Python

Petit dépôt d’**outils Python** autour d’un répertoire **Vindex** déjà produit (par exemple avec **LarQL** / `larql extract`). Il n’y a pas ici d’extracteur Hugging Face : on suppose que vous avez un dossier contenant au minimum `index.json`, les binaires attendus (`embeddings.bin`, `gate_vectors.bin`, etc.) et, pour le forward complet, les poids attention / FFN selon le niveau d’extraction.

## Contenu du dépôt

| Fichier | Rôle |
|--------|------|
| `vindex_infer_python.py` | Port Python de l’inférence type `vindex-infer` (Rust) : charge le Vindex en **f16** (mmap), forward **attention + FFN** ou ablation `--forward ffn-only`, logits via **tête LM liée** à `embeddings.bin`. |
| `vindex_infer_ffn_att.py` | Même moteur que ci-dessus, avec en plus l’option **`--attn-meta`** pour afficher un résumé des libellés sémantiques **Option C** (`attn_meta.bin`). |
| `build_attn_semantic_meta.py` | Construit `attn_meta.bin` (et optionnellement `attn_meta_scores.bin`) à partir de `attn_weights.bin` + `embeddings.bin`, et met à jour `index.json` (`attention_metadata`). Nécessite **PyTorch** (accélération GPU possible). |

## Prérequis

- **Python 3.10+** recommandé.
- Un répertoire **Vindex** valide (niveau au moins compatible avec attention si vous voulez le chemin « full » sans sauter l’attention).

## Installation

```bash
pip install -r requirements.txt
```

**Profil minimal** (sans construire `attn_meta`) : `numpy` + `tokenizers` suffisent pour lancer l’inférence avec `--prompt` si `tokenizer.json` est dans le Vindex. Le script `build_attn_semantic_meta.py` exige en plus **PyTorch** (voir `requirements.txt`).

## Exemples d’utilisation

Inférence (IDs explicites ou prompt texte) :

```bash
python vindex_infer_python.py --vindex ./chemin/vers/modele.vindex --token-ids 818,5279,529,7001,563
python vindex_infer_python.py --vindex ./chemin/vers/modele.vindex -p "The capital of France is"
```

Même chose avec résumé **attn_meta** après le forward :

```bash
python vindex_infer_ffn_att.py --vindex ./chemin/vers/modele.vindex -p "Hello" --attn-meta --attn-meta-layer 20
```

Génération des métadonnées attention « Option C » :

```bash
python build_attn_semantic_meta.py --vindex ./chemin/vers/modele.vindex --top-k 20
python build_attn_semantic_meta.py --vindex ./chemin/vers/modele.vindex --cpu
```

Logs : par défaut **INFO** sur stderr ; `-v` = DEBUG, `-q` = WARNING.

## Windows / PowerShell

Les exemples ci-dessus fonctionnent tels quels. Pour des variables d’environnement, utilisez la syntaxe PowerShell (`$env:NOM="valeur"`) plutôt que `NOM=valeur commande`.

## Licence

Apache-2.0 — alignée sur l’écosystème LarQL / Vindex lorsque applicable.
