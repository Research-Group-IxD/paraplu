# Changelog: what broke in testing and how we fixed it

Every fix below came from a real answer in `logs/truth_check.csv` (model gemma3:12b on the team's MacBook). Newest first.

## 16 Nov (evening): CBS comparison fixes, live city pulse and "Match me to a village"

After running the first CBS and Woensel tests on the Mac (16 Nov 13:20 and 14:35), we saved `poc_v5_before_17nov.py` as version A and fixed seven issues before the 17 Nov client meeting and AI and Data Event:

1. **Reactions compare on the same topic and keep their voice:** `reaction_prompt` and `reaction_on_topic` now allow the second village to compare its own figures on the same topic (against the Eindhoven average) and `keep_fact_sentences` keeps at most one short in-character sentence without an id (no digits, no proper names, no false corrections).
2. **No false corrections across village numbers:** told the reaction prompt that the two villages have different numbers for their own districts, and blocked unbacked `"X is mistaken"` / `"too high"` / `"off"` claims and reactions with numbers not in the guest's facts.
3. **Sentence-level citation cleanup (`clean_sentence_citations`):** moves `[id]` off short `"Yes, there are [G14]."` openers onto the fact sentence, strips ids from `"I don't know"` / refusal / opinion filler, drops stray extra ids whose numbers are absent from the sentence, and drops sentences with invented website domains.
4. **Fun-fact and persona-tic rules:** when asked for a fun fact, the village picks a colourful specific fact (`G05`, `W09`) instead of row 1 (`G01`, `W01`), and `persona.md` files were toned down so Gestel and Woensel do not repeat their river or "last few farms" tic in every answer.
5. **Comparable data across all three villages:** added Woensel CBS 2023 rows (`W12` to `W19`), removed Strijp's duplicate population row (`S10`, keeping `S18` at 39,200), added source link for `W05`, and added everyday market/park/transport facts (`G21`, `S25`, `W20`).
6. **Live city pulse (`L-PULSE`, `data.eindhoven.nl`):** every village now fetches its public playground count (`speelplekken`) and its total and newest BuitenBeter citizen street reports (`meldingen-openbare-ruimte-binnenbeter`) live from Eindhoven's open data portal.
7. **Interactive "Match me to a village" mode (`--match` in `poc.py`, `/match` on the demo page):** a visitor describes who they are or clicks a preset (Student, Young family, New in Eindhoven), and all three villages (`Gestel`, `Strijp`, `Woensel`) pitch why their side of the city fits or does not fit using their own sourced facts, followed by one-click buttons to keep talking to the village the visitor picks.

Side-by-side before/after results on the 10-question event batch are saved in `logs/before_after_17nov.md`.

## 16 Nov: Woensel added, with three new live facts

Woensel (the Oude Toren, 11 facts from Mai's research in the Village entities doc) is the third village. New live fact `L-AIR`: the latest hourly NO2 and O3 values of the Luchtmeetnet station on the Genovevalaan (NL10236), which stands in Woensel itself. New live fact `L-DUST`: PM2.5 and PM10 from the city's own fine-dust sensors (data.eindhoven.nl, real-time-fijnstof-monitoring), for every village in `LIVE_PLACES`: the average of the sensors within 1.5 km of the village's point, or else the nearest one within 3 km. New live fact `L-WEATHER` for every village: sky, temperature, feels-like temperature and wind right now, from Open-Meteo. The air facts give numbers only, no "good" or "bad", because no checked fact says where the line is.

Not tested with the model yet: the files load and the checks run, but nobody has run `questions/woensel.txt` on the Mac, and the three new live calls were only tested against sample answers. "Near the village" is a circle around one point, not the real district border, and neighbouring villages can share a sensor. Still open for Woensel: CBS 2023 figures like G13 to G20, a source link for W05 and the exact page for W02, and facts for the everyday questions the persona promises (bus, market day, parks).

## 8 Nov: truth check of all 123 logged answers

Every answer in `logs/truth_check.csv` (26 Oct to 7 Nov, gemma3:12b) now has a verdict. Claude proposed a verdict and a reason for each answer from the fact lists, the auto notes and this changelog; Nikita spot-checked them. Verdicts describe what the visitor saw: true, partly (true core plus an invented or off-topic part), false, refused well (said it does not know, or PASS), blocked ok or blocked wrongly (a reaction stopped by the topic check).

| Phase | Answers | Correct, refused or blocked well | False | Right language |
| --- | --- | --- | --- | --- |
| Gestel alone, 26 to 27 Oct | 25 | 92% | 0 | 24/25 |
| First two-village runs, 29 to 30 Oct | 16 | 69% | 1 | 8/16 |
| 2 Nov after round 1 | 22 | 82% | 1 | 22/22 |
| 2 Nov after round 2 | 18 | 56% | 2 | 18/18 |
| 2 Nov after round 3 | 12 | 75% | 2 | 12/12 |
| 2 Nov after round 4 | 6 | 83% | 0 | 6/6 |
| 2 Nov client demo with Coen | 14 | 93% | 1 | 14/14 |
| 7 Nov after the stability fix | 10 | 90% | 0 | 10/10 |

What it shows: the language fix worked (every answer in the right language from round 1 on); the round-2 conclusion rule made things worse and was removed; the topic check blocked 12 bad reactions but also one good football answer. Citations are still the weak spot: of the 97 answers that carry fact ids, 51 cite correctly, 28 put ids behind small talk and 18 cite a wrong id. Small samples and different question sets per phase, so this is a trend, not a benchmark.

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
- The truth check (8 Nov) is one pass by Claude with a spot-check; the next round should be marked by a teammate who did not write the prompts.
