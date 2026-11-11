# PedalFix NLP

PedalFix classifies short bike-share maintenance notes into four repair categories. The experiment asks whether word features, character features, or a neural network handle spelling mistakes and text-message abbreviations better.

| Category | Example issue |
| --- | --- |
| Brake | The lever feels loose and the bike takes longer to stop. |
| Drivetrain | The chain skips when the rider pushes hard on the pedals. |
| Tire | The rear tire keeps losing air after inflation. |
| Lock | The station accepts the return but the latch never closes. |

The project uses Python, scikit-learn, and PyTorch. All three models learn from labeled text and predict one of the four categories.

## 1. Understand the data

`data/notes.csv` is included in the repository. The data is synthetic and contains no rider information. The training set combines 72 original complaint phrases with 12 neutral sentence frames. Validation and test notes are separately written and contain new phrasing.

| Split | Rows | How it is used |
| --- | ---: | --- |
| Training | 864 | Learn TF-IDF features, logistic regression weights, character vocabulary, and CNN weights |
| Validation | 24 | Choose the CNN checkpoint with the highest macro-F1 |
| Test | 48 | Evaluate the models on the same notes in clean and noisy form |

Each test row has the original note and a second version generated with abbreviations and adjacent-letter swaps. For example, “It changes gears by itself on the climb.” becomes “it changes gaers by itself on the climb.” A fixed random seed keeps the test variants repeatable. Both versions share the same label and count as one test pair. Text normalization already handles case and punctuation in both conditions.

## 2. Set up the project in VS Code on Windows

1. Download and extract the ZIP.
2. In VS Code, choose **File → Open Folder** and open the inner `pedalfix-nlp` folder. It should directly contain `train.py` and `README.md`.
3. Choose **Terminal → New Terminal**. Run the commands below one line at a time in PowerShell. Use Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

| Command | What it does |
| --- | --- |
| `py -3.12 -m venv .venv` | Creates a separate Python environment for this project. |
| `... -m pip install -r requirements.txt` | Installs scikit-learn and joblib into the project environment. |
| `... -m pip install torch ...` | Installs the CPU build of PyTorch for the neural network. |

If PowerShell says it cannot find `py`, install Python 3.12 or run `python --version` to check whether Python 3.12 is already available through the `python` command. If it is, use `python -m venv .venv` for the first line. The remaining commands call the environment's Python directly.

## 3. Generate the CSV and train

```powershell
.\.venv\Scripts\python.exe dataset.py
.\.venv\Scripts\python.exe train.py
```

- `dataset.py` rebuilds the included CSV from the source phrases and rules. Open `data/notes.csv` in VS Code to see the text, category, split, and noisy counterpart.
- `train.py` fits the two logistic regression baselines and trains the CNN for 12 epochs. Each epoch is one pass through the training rows. The script prints the loss and validation macro-F1 after each epoch, restores the best validation checkpoint, and then evaluates all three models.

| Model | How it works |
| --- | --- |
| Word TF-IDF + logistic regression | Counts single words and word pairs, downweights patterns common across documents, and learns weights that link those features to repair categories. |
| Character TF-IDF + logistic regression | Applies the same idea to groups of 3–5 characters. A misspelled word can still share several fragments with its original spelling. |
| Character CNN | Converts characters into trainable vectors. Convolution filters scan groups of 3, 5, and 7 characters. Max pooling retains the strongest pattern matches before the final layers predict a category. |

Read these files in order to follow the code.

1. `dataset.py` shows where the notes and labels come from and how typos are made.
2. `models.py` shows normalization, TF-IDF pipelines, character encoding, and the CNN.
3. `train.py` shows what the models learn from, how the best CNN epoch is selected, and how scores are calculated.
4. `predict.py` reloads the saved models and applies them to a new note.

## 4. Read the results

These scores come from one CPU run with Python 3.12, scikit-learn 1.8.0, and PyTorch 2.15.0. The CNN checkpoint came from epoch 6. Scores can vary with the PyTorch version and platform.

| Model | Clean macro-F1 | Noisy macro-F1 | Clean accuracy | Noisy accuracy |
| --- | ---: | ---: | ---: | ---: |
| Word TF-IDF + logistic regression | 0.897 | 0.810 | 0.896 | 0.812 |
| Character TF-IDF + logistic regression | 0.958 | 0.875 | 0.958 | 0.875 |
| Character CNN | 0.813 | 0.835 | 0.812 | 0.833 |

Accuracy is the fraction of notes classified correctly. Macro-F1 averages the F1 scores of the four categories equally. F1 combines precision, how often a predicted category is correct, and recall, how many notes in that category the model catches.

Character TF-IDF scored highest on both versions. In test row `test-drivetrain-2`, the word model changed its prediction from drivetrain to lock after the note was corrupted. Character TF-IDF still predicted drivetrain. In `test-drivetrain-4`, the CNN correctly routed “it feles stuck in the hardest gear even after i twist the sihfter” to drivetrain. Both baselines predicted lock.

The CNN got 39 clean notes and 40 noisy notes right. That one-note difference on a small synthetic set is too small to conclude that adding typos improves CNN performance. The training rows reuse 72 core complaints, and the test set has 12 examples per category. Real bike-share notes could use different wording, contain several issues in one note, or fall outside the four categories.

| Output in `results/` | What to look for |
| --- | --- |
| `metrics.csv` | Accuracy and macro-F1 for each model and condition |
| `predictions.csv` | Each test pair, its correct label, and all six predictions |
| `evaluation.json` | Per-category metrics and confusion matrices. Rows are actual labels, and columns are predictions. Both use the listed label order. |
| `baselines.joblib` and `character_cnn.pt` | Trained models used by the prediction script |

## 5. Predict a category from your own note

```powershell
.\.venv\Scripts\python.exe predict.py "The rear tire keeps losing air after I pump it up."
.\.venv\Scripts\python.exe predict.py "frnt brkae leevr wont come back"
```

The first note predicted `tire` with all three models in the sample run. The second predicted `drivetrain` with word TF-IDF and `brake` with both character models. To practice error analysis, try writing a note that avoids words such as “brake,” “tire,” or “lock” and check which model still gets it right.

## 6. Upload to GitHub

1. On GitHub, create a new repository named `pedalfix-nlp`. Leave GitHub's README, .gitignore, and license options unchecked because you are uploading the existing project files to an empty repository.
2. Back in the VS Code terminal, make sure you opened the folder containing `train.py`. Run each command below. Replace `YOUR_USERNAME` with your GitHub username.

```powershell
git init
git add .
git status
git commit -m "Add PedalFix NLP model comparison"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/pedalfix-nlp.git
git push -u origin main
```

| Command | What it does |
| --- | --- |
| `git init` | Starts tracking the local project with Git. |
| `git add .` | Stages the source files, data, and README. The .gitignore excludes the Python environment and training outputs. |
| `git status` | Shows what will be included before you commit. |
| `git commit -m "..."` | Saves a snapshot of the staged files. |
| `git branch -M main` | Names your local branch `main`. |
| `git remote add origin ...` | Connects your local folder to the GitHub repository URL. |
| `git push -u origin main` | Uploads the commit and connects the local branch to GitHub for future pushes. |

3. Refresh the GitHub repository page. Confirm it shows `README.md`, `train.py`, `predict.py`, and `data/notes.csv`. GitHub renders this README on the repository home page.

## Explain the experiment in an interview

“My project compares three ways to classify bike-share repair notes. I used a labeled synthetic dataset and kept the test notes separate from training. I created a typo-heavy copy of each test note and evaluated the same models on both versions. Character-level TF-IDF performed best overall, while the CNN handled some misspelled examples the baselines missed. I reviewed the individual predictions to see which repairs were being confused.”
