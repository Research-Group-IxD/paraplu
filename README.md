# Six Villages, One City: proof of concept

Eindhoven grew out of six villages that merged on 1 January 1920. In this proof of concept two of them talk: **Gestel** (voiced by the Genneper watermill) and **Strijp** (voiced as footballer Coen Dillen). Each village answers only from its own list of sourced facts, shows which facts it used, and a second village can join in with a fact of its own. A local language model plays both, so there is no API key and no cost.

Moonshot AI team (Fontys, AI for Society minor) for the Lectoraat Immersive Technologies. Student prototype, not the real places or people.

## What you need

- A Mac (or any computer) with Python 3.9 or newer. No pip installs.
- [Ollama](https://ollama.com) with the model `gemma3:12b` (8.1 GB). On a laptop with 16 GB memory, close heavy apps first. Too slow? Use `gemma3:4b` (3.3 GB) and add `--model gemma3:4b` to every command.

```
ollama pull gemma3:12b
```

## Run it

| What | Command |
| --- | --- |
| Demo page in the browser (open http://localhost:8000) | `python3 demo_page.py` |
| Talk to one village in Terminal | `python3 poc.py gestel` |
| Two villages: Gestel answers, Strijp may join in | `python3 poc.py gestel --with strijp` |
| Run a question list (every answer is logged) | `python3 poc.py gestel --with strijp --batch questions/demo-check.txt` |
| Force a language | add `--lang nl` or `--lang en` |
| Mac slow or freezing? Health check | `python3 doctor.py` |

Type `quit` to stop a Terminal chat, Ctrl+C to stop the page. Both free the model's memory when they stop.

## How one question travels

1. The code decides the language (English or Dutch) by counting typical words in the question.
2. It builds one message for the model: the village's character (`villages/<name>/persona.md`), eight rules, the village's facts with their ids (`villages/<name>/facts.csv`, like `[G05] Vincent van Gogh painted the Genneper watermill four times in 1884.`), one live fact (rain of the last 7 days from Open-Meteo), and the question.
3. The model answers and puts fact ids behind its sentences.
4. The code checks the answer (see the table below), shows the facts it used with their source links, and saves it in `logs/truth_check.csv`.
5. The second village gets the question and the first village's answer (never its facts) and may speak only if one of its own facts answers the question. Otherwise it passes.

## Checks that catch mistakes

| Check | What it catches | What you see |
| --- | --- | --- |
| Fact-id check | Ids the village doesn't own, like `[L12]` | Red box: "Cites X, which is not in ... facts" |
| Number check | Numbers in none of the village's facts, like an invented "'44" | Red box: "Mentions 44 ... probably made up" |
| Language check | An answer in the wrong language | Asks the model again once; note in the log |
| Own-fact rule | A reaction without one of the second village's own facts | Counts as a pass |
| Copy filter | Sentences copied from the first village, "I don't know" in a reaction | Removed |
| Fact-sentence filter | Sentences in a reaction with no fact behind them ("Gestel is mistaken.") | Removed |
| Topic check | A reaction about something else than the question (a yes/no question to the model) | Counts as a pass |
| Truth log | Everything | Every answer in `logs/truth_check.csv` for a hand check (verdict: true, partly, false, refused well) |

What these checks can't catch: an id put behind a sentence that the fact doesn't say, and invented details without numbers. That is what the hand check is for. The history of every problem and fix is in [docs/CHANGELOG.md](docs/CHANGELOG.md).

## Add your village

1. Copy `villages/_template` to `villages/<your-village>` (lowercase, no spaces, for example `woensel`).
2. Write `persona.md`: character only, no facts. Notes written `<!-- like this -->` are for you; the model never sees them.
3. Fill `facts.csv`: one fact per row with an id (W01, W02, ...) and a source link.
4. Optional live data: add the village's coordinates to `LIVE_PLACES` in `poc.py`.
5. Test it: `python3 poc.py <your-village>`. No other code changes needed.

## Files

| Path | What |
| --- | --- |
| `poc.py` | The shared brain: prompts, rules, language, checks, log |
| `demo_page.py`, `web/index.html` | The demo page (Python standard library only) |
| `doctor.py` | Health check when the Mac gets slow |
| `villages/` | One folder per village: `persona.md` + `facts.csv` |
| `questions/` | Question lists for batch tests, with expected answers as comments |
| `docs/CHANGELOG.md` | What broke in testing and how it was fixed |
| `logs/` | Answer log and doctor reports (not in Git: visitors' questions stay local) |

## Status (7 Nov 2026)

Tested on a MacBook Pro M2 Pro (16 GB) with gemma3:12b during four test rounds on 2 Nov and in the client demo the same day. Known limits: the 29 facts are not double-checked yet, the villages don't have the same kinds of facts (so comparisons go wrong), and two villages take about 20 seconds per question.

## Me & My Machine label

Cyborg. The concept, the village facts and the test questions come from the team; the code was written with Claude (Anthropic) from our design, every change was run and tested on the Mac, and each fix came from a real test log. (Confirm or change the level before handing in.)
