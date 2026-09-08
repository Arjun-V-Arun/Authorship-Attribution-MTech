# Where the project stands — assessment, findings, and what to do next

**1 September 2026.** Written against the results in `comparison.md`,
`baseline_ladder_all.md`, and the six `low-baselines*.txt` files.

Tags: **[E]** established by your own runs · **[L]** established in the
literature · **[I]** my inference · **[S]** speculative.

---

## Part 1 — Is this a paper?

**Yes, and it is closer to done than you think. But it is a specific kind of
paper, and mistaking which kind is the main risk.**

You do not have a methods paper. You have not invented a model, and nothing in
your results beats a strong baseline by an interesting margin. If you submit as
though the contribution were modelling, a reviewer will correctly say that a
linear SVM on character n-grams is thirty years old.

What you have is a **resource-and-evaluation paper with a methodological
finding attached**, and in that genre it is genuinely competitive. The
components:

| Asset | Status | Strength |
|---|---|---|
| First Malayalam AA corpus, ~700 works, ~28 authors | built | high — nothing comparable exists |
| Author-biography contamination, discovered and quantified | measured | **high — this is your headline** |
| Protocol robustness: honest number unmoved by the leak | measured | high |
| Content-free stylometry: 0.76–0.77 from 100 word types | measured | high |
| Model ladder: Delta underperforms char/word features | measured | medium-high |
| Length-requirement curve | measured | medium |
| sklearn tokenisation breaks Malayalam | measured | medium, very citable |
| Cross-genre attribution | **not possible with this corpus** | — |
| Transformer baselines | **missing** | this is the gap reviewers will name |
| Second-language replication | **missing** | this is what would make it Q1 |

**Realistic venue today:** ACM TALLIP (Q2, and it published the Urdu AA paper),
or an LREC-family conference. **[L]**

**Realistic venue after two more experiments:** Language Resources and
Evaluation (Q1 in Linguistics & Language). The two experiments are transformer
baselines and a Hindi replication. Neither requires new data. **[I]**

**Not realistic:** ACL/EMNLP main, TACL. Those want a methodological
contribution you do not have and should not fake.

---

## Part 2 — Every step, and what it meant

### Step 1. Corpus construction from Sayahna TEI

**Reasoning.** Malayalam has no AA dataset. Sayahna publishes TEI-encoded
digital editions under Creative Commons, with structured author metadata —
born-digital, so no OCR noise, and legally redistributable.

**Expectation.** A few hundred works across a dozen authors.

**Finding [E].** 127 fiction works (13 authors) and 620 essays (17 authors).
One work = one document, so chunk-level leakage is impossible by construction.

**Meaning.** The design decision that matters is the unit of analysis. The
standard practice in literary AA — chunk a novel, split the chunks randomly —
puts passages of one work on both sides of the train/test boundary. Your corpus
cannot do that. It is a small decision that removes the single most common
failure mode in this literature. **[I]**

### Step 2. Standard baseline

**Expectation.** Something in the 0.85–0.95 range, in line with published Indic
work.

**Finding [E].** Fiction 0.963, essays 0.978 macro-F1.

**Meaning.** You were right to distrust it. In AA, a high number on a small
closed author set is the default outcome, not evidence of a good system. The
appropriate response to 0.978 is suspicion, and that instinct is what produced
everything below.

### Step 3. Diagnostic ablations

**Reasoning.** "No chunk leakage" is much weaker than "measuring style." A
model can reach 0.98 via named entities, topic, or digitisation provenance.

**Expectation.** Masking author-unique tokens would cost a lot if the signal
were topical.

**Finding [E].** In the essay corpus, every condition returned an identical
0.978 — masking 86% of token types changed nothing.

**Meaning.** Not robustness. Saturation. At 0.993 accuracy roughly four
documents out of 610 were wrong, and the same borderline ones failed under
every transformation. **A task with no headroom cannot be ablated.** This is a
transferable lesson: run ablations at the hard operating point, not the easy
one. **[I]**

### Step 4. Two bugs found by reading output rather than trusting it

**Finding [E], both verified directly:**

- `char_wb` n-grams never cross a word boundary, so shuffling word order
  produces a byte-identical feature matrix. The word-order test was vacuous by
  construction, not a result.
- sklearn's default word pattern `\b\w\w+\b` destroys Malayalam. Vowel signs
  and virama are combining marks that Python's `\w` does not match:
  `സഞ്ജയൻ → ['സഞ','ജയൻ']`, and `ഭാഷാശാസ്ത്രം → []` — dropped entirely.

**Meaning.** The second is publishable on its own terms. Any Indic NLP work
using sklearn word features on Malayalam has silently been discarding tokens,
and nobody appears to have reported it. It is a one-paragraph contribution that
costs you nothing and will be cited. **[I]**

### Step 5. The author-biography leak

**Reasoning.** Once the tokeniser was fixed, the feature lists became readable —
and showed authors' own names, birth years, awards, and the literal words
*biography* and *biographical note*.

**Finding [E].** Sayahna appends an author note to the body of each work.
Karassery had three separate bio paragraphs in 124 of 125 works, including his
wife's and children's names. Rajalakshmi had seven, in all 14 works. Only
1.7–5.4% of words, but present in 96 and 439 works respectively.

**Meaning.** A short paragraph appearing in every one of an author's works and
nobody else's is a perfect label. Size was never the issue. Removing it cost
0.070 (fiction) and 0.033 (essays) of headline accuracy. **[E]**

### Step 6. The protocol robustness result

**Finding [E].** The combined honest protocol — balanced, entity-masked,
length-equalised — returned **0.752 / 0.752 / 0.752** across raw, clean and
strict text in fiction. Identical to three decimals, while the standard
protocol moved by 0.070.

**Meaning. This is your strongest methodological claim.** The reason is
structural: bio text is by definition author-unique, so entity masking removed
it before you knew it was there. **A leakage-controlled protocol was immune to a
corpus defect that the field's standard protocol was not.** That is a stronger
argument for the protocol than any prose could make, because it is a natural
experiment you did not design. **[I]**

### Step 7. Content-free stylometry

**Finding [E].** With only the 100 most frequent word types and everything else
masked: fiction 0.762, essays 0.773. Against chance of 0.083 and 0.062 — roughly
9× and 13× chance.

**Meaning.** Malayalam authorial style is genuinely detectable with no content,
no names, no topic. This is Mosteller and Wallace reproduced in Malayalam, and
I can find no equivalent demonstration for any Dravidian language. **[I]**

### Step 8. The model ladder

**Expectation (mine, stated in advance).** Character n-grams would beat word
n-grams, because Malayalam encodes in bound morphology what English carries in
free function words.

**Finding [E].**

| | fiction | essays |
|---|---|---|
| char 3–5 gram | 0.914 | 0.905 |
| word 1–2 gram | 0.912 | **0.935** |
| char 2–4 / 4–6 | within 0.010 of char 3–5 | same |
| Burrows' Delta | 0.784 | 0.633 |
| stylometric + LogReg | 0.794 | 0.781 |
| NearestCentroid (char) | 0.751 | 0.850 |

**My prediction was wrong.** Word units tie in fiction and *win* in essays.

**Meaning, corrected.** The earlier `char_wb` > `char` result (+0.015) supports
a narrower claim than I extrapolated. It says n-grams should not cross word
boundaries. It does **not** say the signal is sub-lexical.

The defensible statement: **the author signal in Malayalam sits at and within
the word, not across words.** Lexical choice and word-internal structure both
carry it; cross-word phrasing carries little. That is still a contrast with
English stylometry, where function-word sequence matters. It is just a smaller
claim than I had you building toward. **[E/I]**

**On Burrows' Delta.** The NearestCentroid row is the control that makes this
interpretable. In fiction, Delta (0.784) actually beats NearestCentroid (0.751),
so the gap to the SVM is about discriminative training, not features. In essays,
Delta collapses to 0.633 while NearestCentroid on character features reaches
0.850 — same paradigm, same absence of training, different features.

So the claim is narrow and evidenced: **for Malayalam non-fiction across 16
authors, most-frequent-word features are substantially weaker than character
features, independent of classifier.** Do not claim more. **[E]**

**On the stylometric row.** 0.794 / 0.781 from ~30 readable features, beating
Delta in both corpora. Hand-built surface statistics outperforming the canonical
function-word method is a clean, quotable result, and it makes the paper
interpretable rather than a black box. **[E]**

### Step 9. Discarded results

Two things you should not report, and knowing why matters:

- **Multinomial Naive Bayes, 0.223 / 0.228.** No `class_weight` parameter, so
  on a 145-to-4 imbalance the priors swamp the likelihood. A method artifact, not
  a fact about Malayalam. Replaced with ComplementNB.
- **The cross-genre experiment.** After deduplication only Vallathol has ≥4
  works in both genres. The corpus is two disjoint populations: the fiction list
  is story writers, the essay index is critics and columnists. I pushed this as
  your strongest missing result for several turns and was wrong.

---

## Part 3 — How your work compares to Indic AA

### What exists

| Language | Best venue reached | Authors | Genre | Methods | Leakage control |
|---|---|---|---|---|---|
| **Bengali** | journal + sustained arXiv line (Das & Mitra 2011; Chakraborty 2012; Pal & Chakraborty 2012; Chowdhury 2018) | 2–6 (Tagore, Sarat Chandra) | novels, news, blogs | char n-grams, vocabulary richness, word embeddings | none reported |
| **Urdu** | **ACM TALLIP 21(3), 2021** (Nazir et al., "Authorship Attribution for a Resource Poor Language—Urdu") | — | news/columns | feature engineering + ML | none reported |
| **Telugu** | JATIT; Springer AISC 343 | ~5 | editorial articles | POS, function words (*avyayas*), char/word n-grams | none reported |
| **Kannada** | Springer LNNS (Chandrika & Kallimani 2022) | small | — | char + word n-gram amalgamation | none reported |
| **Punjabi** | conference | 5 poets | poetry | J48 selection → SVM/NB | none reported |
| **Hindi** | Research Square preprint (Anand et al. 2024); IITK CS365 2014 course project | 5 | short stories, books | n-grams, TF-IDF vs fine-tuned mBERT | none reported |
| **Tamil** | no dedicated modern AA found | — | — | — | — |
| **Malayalam** | **none** | — | — | — | — |

**[L]** for all rows above.

### How these papers are written

A recognisable template:

1. Curate 3–10 authors from a digital library or newspaper archive.
2. Extract classical features — character n-grams, word n-grams, POS,
   vocabulary richness, function-word frequencies.
3. Run a bank of classifiers, often through WEKA: SVM, Naive Bayes, J48,
   Random Forest, sometimes an LSTM or BERT.
4. Report accuracy from a random split or plain k-fold.
5. Headline 85–95%.
6. No corpus release, no error analysis, no confound discussion.

Venues run from non-indexed journals up to Springer chapters, with **one**
TALLIP paper as the ceiling. **[L]**

### Three observations worth putting in your related-work section

**1. Nobody reports leakage-controlled evaluation.** Across every language
above, splits are random and topic is uncontrolled. Your finding that author
biographies were embedded in the source text is directly relevant to all of
them — Bengali, Telugu and Kannada work drew on the same kind of digital-library
sources, which commonly attach author notes. You cannot recompute their
results, and should not claim they are wrong. You *can* show on a comparable
corpus that the reported figure moves by 0.03–0.07 under a control they did not
apply. That is a claim about interpretability, not about any individual paper.
**[I]**

**2. The Telugu contrast is the most interesting comparison available to you.**
Telugu is also Dravidian and also agglutinative, and that work reported
function-word features (*avyayas*) performing well, above 85%. Your Burrows'
Delta — a most-frequent-word method — reaches only 0.633 on essays. Two
Dravidian languages, apparently opposite conclusions about function-word
stylometry. The likely explanation is task difficulty rather than linguistics:
5 authors versus 16, and a topically homogeneous single-genre set. **But you can
say this precisely, with the NearestCentroid control to separate features from
classifier, and no one else has.** **[I/S]**

**3. Your scale is already competitive.** 12–16 authors and 114–599 works is
larger than nearly all of the above, which cluster at 5 authors. Combined with
the protocol, this is the strongest Indic AA evaluation setup published. **[I]**

### Where you would sit

| | typical Indic AA paper | yours |
|---|---|---|
| Authors | 5 | 12–16 |
| Works | 50–150 | 114–599 |
| Splits | random | work-disjoint, stratified k-fold |
| Metric | accuracy | macro-F1 with fold variance |
| Confound analysis | none | Cramér's V across 5 provenance variables |
| Leakage audit | none | quantified, with before/after |
| Sanity floors | none | label permutation, length-only |
| Corpus released | no | yes (CC BY-SA) |
| Novel model | sometimes claimed | **none** |

The last row is the honest weakness. Everything above it is your case.

---

## Part 4 — What to do now

Ranked by contribution per unit of effort.

### A. Hindi replication — highest leverage

You already have a Hindi corpus and working baseline code. Run the identical
protocol: same splits, same ladder, same diagnostics.

Why it matters more than anything else on this list: it converts a case study
into a claim about language families. Malayalam is Dravidian and agglutinative;
Hindi is Indo-Aryan and fusional. **A finding that holds in one and breaks in the
other is more interesting than one that holds in both.** Specifically, the
word-versus-character result and the Delta result are exactly the kind of thing
that should differ by morphological type.

This is the single change that moves you from a Q2 target to a plausible Q1
target. Cost: about a week, no new data. **[I]**

### B. Transformer baselines — required, not optional

MuRIL, IndicBERT v2, and XLM-R base, all fitting your RTX 4070. Every reviewer
will ask, and the 2024 Hindi paper's finding that traditional methods beat
mBERT gives you a specific hypothesis to test rather than a box to tick.

Handle the context-length problem explicitly: 512 subword tokens is roughly
250–350 Malayalam words because of tokeniser fertility, so a transformer sees
about a tenth of a story. Segment and aggregate, and report that as a design
decision. It is also a legitimate reason transformers may lose. Cost: a week.

### C. Cross-topic experiment — the domain-shift result you can actually run

Cross-genre is dead, but seven essayists have 18+ works each (Kuttipuzha 144,
Karassery 125, Rajeswari 92, CJ Thomas 64, Kesari 46, Sanjayan 29,
Madhusudhanan 18) — 518 works, chance 0.143. Cluster each author's essays by
content and hold out whole topic clusters, so test topics are unseen in
training.

This is a recognised hard setting in the PAN lineage and no Indic study has run
it. Check first that the authors' essays span varied topics; Rajeswari's
political columns may be too narrow. Cost: 1–2 days. **[L/I]**

### D. Datasheet and release

Sources, licences, the pseudonym issue (K. Rajeswari is A. Jayasankar's pen
name, 92 works), exclusions, the bio-stripping procedure, known confounds.
Zenodo DOI. Half a day, and it is what makes an LRE submission credible.

### E. Author verification

Reframe as "did this author write this, yes or no?" instead of "which of these
16?" This is where the field's attention actually is — PAN 2026 runs five tasks
and none is closed-set attribution. Your corpus supports it without new data.
Good thesis material for early 2027; not needed for the December submission.
**[L]**

### The quantum thread

Stage 1 gives you a validated ~30-dimensional interpretable stylometric feature
vector that reaches 0.79 macro-F1 on its own. That is a genuinely plausible
quantum-kernel setting: low dimension, small sample, no gradient-hungry
architecture — the regime where quantum kernels are competitive at all. Frame
it as characterising where quantum kernels do and do not help for stylometric
classification, not as beating classical baselines. An honest negative result,
properly analysed, is publishable; an overclaimed positive is not. **[S]**

---

## Part 5 — The paper, restructured around what you actually have

**Title:** *MalAA: A Leakage-Controlled Corpus and Evaluation Protocol for
Malayalam Authorship Attribution*

**Claim 1 (resource).** First Malayalam AA corpus. ~700 works, ~28 authors,
TEI-derived metadata, CC-licensed, released with splits and code.

**Claim 2 (methodological, the headline).** Author biographies embedded in
source text inflate standard evaluation by 0.03–0.07 macro-F1. A
leakage-controlled protocol is invariant to this defect while the standard
protocol is not, demonstrated on the same corpus before and after cleaning.

**Claim 3 (empirical).** Malayalam authorship is detectable from content-free
features: 0.76–0.77 macro-F1 from the 100 commonest word types alone, against
chance of 0.06–0.08.

**Claim 4 (linguistic).** The signal sits at and within the word, not across
words. Most-frequent-word stylometry (Burrows' Delta) underperforms character
and word features substantially in Malayalam non-fiction, and the
NearestCentroid control shows this is a property of the features, not the
classifier.

**Claim 5 (practical).** Standard NLP tooling silently corrupts Malayalam word
tokenisation, discarding tokens entirely.

**Limitations, stated up front:** closed set of 12–16 authors; single
digitisation source; author and era partly confounded; no cross-genre
experiment possible with this corpus, and why; no morphological analyser in the
pipeline; entity masking is a proxy for NER, which Malayalam lacks at quality.

That is a complete paper. It needs B (transformers) to survive review, and A
(Hindi) to reach Q1.

---

## Part 6 — The two corrections I owe you

1. **The sub-lexical claim was overstated.** I extrapolated a +0.015 within-word
   advantage into "the author signal lives below the word." The ladder shows word
   units matching or beating character units. The corrected claim is at-and-within
   the word. If you had written the stronger version, a reviewer would have found
   this with one experiment.

2. **Cross-genre was never feasible with this corpus.** I checked author overlap
   and not work overlap, and pushed it as your strongest missing result across
   several turns. Track C (cross-topic) is the replacement and is genuinely
   runnable.

Both were caught by experiments that were designed to be capable of
contradicting the hypothesis. Keep designing them that way.
