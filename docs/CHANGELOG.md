# Changelog: what broke in testing and how we fixed it

Every fix below came from a real answer in `logs/truth_check.csv` (model gemma3:12b on the team's MacBook). Newest first.

## 7 Nov: the Mac froze after the client demo

**Problem.** On the evening of 2 Nov, answers took minutes and never arrived, the Mac lagged and the sound crackled.
**Likely cause.** The model's context size was left to Ollama's own setting, and the model was kept in memory for 60 minutes. A large context makes gemma3:12b too big for the GPU on a 16 GB Mac, so it spills onto the CPU and the Mac starts swapping.
**Fix.** Every call now sets `num_ctx` 4096 and a maximum answer length (`MODEL_OPTIONS` in `poc.py`); the model stays loaded 20 minutes instead of 60 and is unloaded when the chat, batch run or page stops; after 120 seconds without an answer you get a clear message. New `doctor.py` reports memory, leftover processes, what Ollama has loaded (and how much runs on the GPU) and the real speed.

## 2 Nov: four fix rounds before the client demo (14:00)

| Round | Problem in the log | Example | Fix | Result |
| --- | --- | --- | --- | --- |
| 1 | Wrong language | 30 Oct: "Were there bombings here during the war?" answered "Ik weet het niet...". At least 5 of 33 answers in the wrong language | The code counts Dutch and English words in the question, tells the model the language and asks again once if wrong. The old rule named "Dutch" and made the model lean Dutch | 12:28 test: 24 of 24 answers and reactions in the right language |
| 1 | Second village copied the first | "Ik weet het niet [Gestel]. Strijp werd..."; 4 of 8 reactions cited ids they don't own | First answer is shown without ids; copied sentences and "I don't know" removed; a reaction needs one of its own facts | 12:28: 0 copies, 0 wrong ids in 7 reactions |
| 1 | Bracket noise | "[I don't know]" | Only real fact ids allowed in brackets | Gone |
| 1 | Rain told twice | Both villages read out the same rain total | The second village doesn't get a live fact the first already used | Gone |
| 2 | Off-topic additions | "Who painted you?" answered by Strijp: "I was born in Strijp" | The second village may only speak when a fact answers the question, plus a yes/no topic check by the model | 12:59: 4 off-topic additions stopped, but a good football answer too |
| 2 | Invented details | Gestel "remembers the river rose considerably in '44" (no such fact) | Number check: a number in none of the facts gets a red warning | Caught in every later run |
| 3 | Our own rule backfired | A round-2 rule "end a comparison with a clear conclusion" gave "So, Gestel's timeline is off" and "Strijp predates Gestel" (wrong) | Rule removed; looser topic check ("same topic?") | 13:11: off-topic gone, but two false corrections ("Gestel is mistaken.") |
| 4 | False corrections | "Strijp became a municipality in 1795. Gestel is mistaken." | A reaction keeps only sentences that carry a fact id | In the client demo Strijp's reactions were fact-only |

Also on 2 Nov: demo page with source links, timer and question buttons; the model is kept loaded between questions.

## 29 to 30 Oct: first two-village run

Strijp (Coen Dillen, 17 facts, from Nilay's prototype) joined as the second village. First run worked but showed the problems fixed on 2 Nov: false corrections, copied sentences with made-up ids, language flips, about 17 s per question.

## 26 Oct: first test, Gestel alone

Rules 2, 3, 5 and 8 sharpened: ids only behind sentences a fact supports, no claims about the world without an id, no repeated facts, answer in the visitor's language.

## Still open

- Ids put behind small talk ("dat weet ik niet [G11]") and claims a fact doesn't make: only the hand check catches these.
- The villages hold different kinds of facts (Gestel: when the mill was first mentioned; Strijp: when its name was recorded), so "who is older" goes wrong. Next: the same fact types for every village, and comparable numbers from one source (CBS district figures).
- Two villages take about 20 seconds per question (target 10 s).
- Verdicts in the truth log are not filled in yet, so there are no accuracy numbers.
