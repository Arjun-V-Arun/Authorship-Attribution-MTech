# Authorship Attribution for Indic Languages — Literature Review Corpus

Compiled for the M.Tech project on Malayalam authorship attribution (with a
quantum–classical hybrid component). Assembled September 2026.

**How to read the status tags.** Every entry is tagged:

- `[V]` — **Verified**: I located the paper's own abstract page, PDF, or publisher record.
- `[R]` — **Reference-harvested**: the entry appears in the reference list of a
  verified paper, but I have not opened the paper itself. Bibliographic details
  (page numbers, exact title wording) need checking before you cite it.

Do not cite an `[R]` entry without pulling it first. Several of them are in
venues with weak indexing and the metadata circulating in reference lists is
inconsistent.

**Headline finding.** There is no published, dedicated authorship attribution
study on Malayalam literary prose. Malayalam appears in the AA literature only
as one low-resource column in a multilingual benchmark (Kim et al. 2025, entry
30), evaluated on Wikipedia text, not literature. Every other major Indian
language — Bengali, Urdu, Hindi, Telugu, Kannada, Tamil, Marathi, and even
Assamese — has at least one dedicated paper. This is the gap your project sits
in, and it is a real one. See §8.

---

## 1. Surveys and foundations

The papers everyone cites. Your related-work section needs most of these.

| # | Entry | Status |
|---|---|---|
| 1 | Stamatatos, E. (2009). *A survey of modern authorship attribution methods.* JASIST 60(3), 538–556. doi:10.1002/asi.21001 — The canonical survey. Instance-based vs. profile-based framing, feature taxonomy. | [V] |
| 2 | Koppel, M., Schler, J., & Argamon, S. (2009). *Computational methods in authorship attribution.* JASIST 60(1), 9–26. | [R] |
| 3 | Neal, T., Sundararajan, K., Fatima, A., Yan, Y., Xiang, Y., & Woodard, D. (2017). *Surveying stylometry techniques and applications.* ACM Computing Surveys 50(6), 1–36. doi:10.1145/3132039 | [R] |
| 4 | Lagutina, K., Lagutina, N., Boychuk, E., Vorontsova, I., Shliakhtina, E., & Belyaeva, O. (2019). *A Survey on Stylometric Text Features.* 25th FRUCT Conference. doi:10.23919/fruct48121.2019.8981504 — Most systematic feature inventory available. | [R] |
| 5 | Zheng, W., & Jin, M. (2022/2023). *A review on authorship attribution in text mining.* WIREs Computational Statistics 15, e1584. doi:10.1002/wics.1584 | [R] |
| 6 | Xie, H., Habibi Lashkari, A., Vasudevan, N., & Sharma, D. P. (2024). *Authorship Attribution Methods, Challenges, and Future Research Directions: A Comprehensive Survey.* Information 15(3), 131. doi:10.3390/info15030131 — The most recent broad survey; use this as your "current landscape" anchor. | [R] |
| 7 | Cammarota, V., Bozza, S., Roten, C.-A., & Taroni, F. (2024). *Stylometry and forensic science: A literature review.* Forensic Science International: Synergy 9, 100481. | [R] |
| 8 | Tyo, J., Dhingra, B., & Lipton, Z. C. (2022). *On the State of the Art in Authorship Attribution and Authorship Verification.* arXiv:2209.06869 — Important corrective: argues much reported SOTA does not survive controlled re-evaluation. Directly relevant to how you set up your baselines. | [V] |
| 9 | Savoy, J. (2020). *Machine Learning Methods for Stylometry: Authorship Attribution and Author Profiling.* Springer. doi:10.1007/978-3-030-53360-1 — Book. R code and datasets on the author's GitHub. | [V] |
| 10 | Mosteller, F., & Wallace, D. L. (1963). *Inference in an authorship problem.* JASA 58(302), 275–309. — The Federalist Papers study. Cite for provenance. | [R] |
| 11 | Burrows, J. (2002). *'Delta': a measure of stylistic difference and a guide to likely authorship.* Literary and Linguistic Computing 17(3), 267–287. | [R] |
| 12 | Evert, S., Proisl, T., Jannidis, F., Reger, I., Pielström, S., Schöch, C., & Vitt, T. (2017). *Understanding and explaining Delta measures for authorship attribution.* DSH 32(suppl. 2), ii4–ii16. doi:10.1093/llc/fqx023 | [R] |
| 13 | Eder, M. (2015). *Does Size Matter? Authorship Attribution, Short Samples, Big Problem.* DSH 30(2), 167–182. — Directly relevant to your document-length decisions. | [R] |
| 14 | Eder, M., Rybicki, J., & Kestemont, M. (2016). *Stylometry with R: A Package for Computational Text Analysis.* The R Journal 8(1), 107–121. — `stylo`, the standard toolkit. | [R] |

## 2. Feature engineering, cross-topic and cross-genre robustness

This block matters more than usual for you: your Sayahna corpus splits into
fiction and non-fiction subcorpora, so topic and genre confounds are a live
threat to any result you report.

| # | Entry | Status |
|---|---|---|
| 15 | Peng, F., Schuurmans, D., Wang, S., & Kešelj, V. (2003). *Language Independent Authorship Attribution using Character Level Language Models.* EACL 2003, E03-1053. — The character-LM baseline; unusually well suited to agglutinative languages. | [V] |
| 16 | Kešelj, V., Peng, F., Cercone, N., & Thomas, C. (2003). *N-gram-based author profiles for authorship attribution.* PACLING. | [R] |
| 17 | Sapkota, U., Bethard, S., Montes, M., & Solorio, T. (2015). *Not All Character N-grams Are Created Equal: A Study in Authorship Attribution.* NAACL-HLT 2015, 93–102. — Shows affix and punctuation n-grams carry nearly all the signal. For Malayalam, "affix" is a much bigger category than in English. | [V] |
| 18 | Stamatatos, E. (2013). *On the robustness of authorship attribution based on character n-gram features.* Journal of Law and Policy 21(2). | [V] |
| 19 | Stamatatos, E. (2017). *Authorship attribution using text distortion.* EACL 2017, 1138–1149. — Masks topical content before feature extraction. | [V] |
| 20 | Markov, I., Stamatatos, E., & Sidorov, G. (2018). *Improving cross-topic authorship attribution: The role of pre-processing.* CICLing 2017, LNCS, 289–302. | [V] |
| 21 | Kestemont, M. (2014). *Function words in authorship attribution. From black magic to theory?* CLFL @ EACL, 59–66. | [R] |
| 22 | Argamon, S., & Levitan, S. (2005). *Measuring the usefulness of function words for authorship attribution.* ACH/ALLC. | [R] |
| 23 | Hirst, G., & Feiguina, O. (2007). *Bigrams of syntactic labels for authorship discrimination of short texts.* LLC 22(4), 405–417. | [R] |
| 24 | Sidorov, G., Velasquez, F., Stamatatos, E., Gelbukh, A., & Chanona-Hernández, L. (2014). *Syntactic n-grams as machine learning features for NLP.* Expert Systems with Applications 41(3), 853–860. | [R] |
| 25 | Sari, Y., Stevenson, M., & Vlachos, A. (2018). *Topic or Style? Exploring the Most Useful Features for Authorship Attribution.* COLING 2018, 343–353. | [V] |
| 26 | Altakrori, M., Cheung, J. C. K., & Fung, B. C. M. (2021). *The Topic Confusion Task: A Novel Evaluation Scenario for Authorship Attribution.* Findings of EMNLP 2021, 4242–4256. — Gives you a clean protocol for proving your model is not learning topic. | [V] |
| 27 | Bischoff, S. et al. (2020). *The Importance of Suppressing Domain Style in Authorship Analysis.* arXiv:2005.14714 | [V] |
| 28 | Markov, I., Baptista, J., & Pichardo-Lagunas, O. (2017). *Authorship Attribution in Portuguese Using Character N-grams.* Acta Polytechnica Hungarica 14(3). doi:10.12700/aph.14.3.2017.3.4 | [V] |
| 29 | Sapkota, U., Solorio, T., Montes, M., & Bethard, S. (2016). *Domain adaptation for authorship attribution: improved structural correspondence learning.* ACL. | [R] |

## 3. Neural and representation-learning state of the art (2019–2026)

This is the block that decides what "state of the art" means in your paper. Note
the recurring finding, discussed in §8, that neural methods do not automatically
beat TF-IDF + SVM on low-resource literary corpora.

| # | Entry | Status |
|---|---|---|
| 30 | **Kim, J., Zhang, H., & Jurgens, D. (2025). *Leveraging Multilingual Training for Authorship Representation: Enhancing Generalization across Languages and Domains.* EMNLP 2025, 34867–34892. arXiv:2509.16531** — **The single most important entry for you.** Trains a multilingual authorship-representation model on 4.5M authors across 36 languages. Malayalam is one of the 22 evaluated languages, with only 1,718 authors total (172 in test) — among the smallest. Reported Malayalam R@8: 54.70 (monolingual XLM-R) → 63.95 (multilingual); MRR 37.94 → 45.01. Llama-3.2-1B does far worse on Malayalam (36.31 R@8) because its tokenizer shatters Malayalam into near-bytes. Two methods introduced: Probabilistic Content Masking (PCM) and Language-Aware Batching (LAB). Code: github.com/junghwanjkim/multilingual_aa; model on HuggingFace (`Blablablab/multilingual-style-representation`). **This is your citable Malayalam number, and its domain (Wikipedia) is exactly what your literary corpus is not.** | [V] |
| 31 | Rivera-Soto, R. A., Miano, O. E., Ordonez, J., Chen, B. Y., Khan, A., Bishop, M., & Andrews, N. (2021). *Learning Universal Authorship Representations* (LUAR). EMNLP 2021, 913–919. — The contrastive-learning backbone everyone builds on. | [V] |
| 32 | Zhu, J., & Jurgens, D. (2021). *Idiosyncratic but not arbitrary: Learning idiolects in online registers reveals distinctive yet consistent individual styles.* EMNLP 2021, 279–297. | [V] |
| 33 | Wegmann, A., Schraagen, M., & Nguyen, D. (2022). *Same Author or Just Same Topic? Towards Content-Independent Style Representations* (CAV). RepL4NLP @ ACL, 249–268. | [V] |
| 34 | Sawatphol, J., Chaiwong, N., Udomcharoenchaikit, C., & Nutanong, S. (2022). *Topic-Regularized Authorship Representation Learning.* EMNLP 2022, 1076–1082. | [V] |
| 35 | Wang, A., Aggazzotti, C., Kotula, R., Rivera-Soto, R., Bishop, M., & Andrews, N. (2023). *Can authorship representation learning capture stylistic features?* TACL 11, 1416–1431. | [V] |
| 36 | Patel, A., Zhu, J., Qiu, J., Horvitz, Z., Apidianaki, M., McKeown, K., & Callison-Burch, C. (2025). *StyleDistance: Stronger Content-Independent Style Embeddings with Synthetic Parallel Examples.* NAACL 2025, 8662–8685. | [V] |
| 37 | Qiu, J., Zhu, J., Patel, A., Apidianaki, M., & Callison-Burch, C. (2025). *mStyleDistance: Multilingual Style Embeddings and Their Evaluation.* arXiv:2502.15168 — Note that entry 30 reports it performing poorly in their AA setup; worth reproducing that yourself. | [V] |
| 38 | Huertas-Tato, J. et al. (2024). *STAR: Zero-shot authorship attribution via supervised contrastive representations.* Knowledge-Based Systems 296, 111867. | [V] |
| 39 | Fabien, M., Villatoro-Tello, E., Motlicek, P., & Parida, S. (2020). *BertAA: BERT fine-tuning for Authorship Attribution.* ICON 2020, 127–137. — Published at ICON, i.e. in the Indian NLP community. Natural baseline and a natural venue precedent. | [V] |
| 40 | Barlas, G., & Stamatatos, E. (2020). *Cross-Domain Authorship Attribution Using Pre-trained Language Models.* AIAI 2020, 255–266. | [V] |
| 41 | Shrestha, P., Sierra, S., González, F., Montes, M., Rosso, P., & Solorio, T. (2017). *Convolutional neural networks for authorship attribution of short texts.* EACL 2017, 669–674. | [R] |
| 42 | Boenninghoff, B., Nickel, R. M., Zeiler, S., & Kolossa, D. (2019). *Similarity learning for authorship verification in social media.* ICASSP 2019, 2457–2461. | [R] |
| 43 | Hu, Z. et al. (2024). *Contrastive disentanglement for authorship attribution.* WWW '24 Companion, 1657–1666. | [R] |
| 44 | Man, H., & Nguyen, T. H. (2024). *Counterfactual augmentation for robust authorship representation learning.* SIGIR '24, 2347–2351. | [R] |
| 45 | Fincke, S., & Boschee, E. (2024). *Separating Style from Substance: Enhancing Cross-Genre Authorship Attribution through Data Selection and Presentation.* arXiv:2408.05192 | [R] |
| 46 | *Layered Insights: Generalizable Analysis of Human Authorial Style by Leveraging All Transformer Layers.* arXiv:2503.00958 (v3, 2025) — Extends LUAR and Wegmann by using all transformer layers rather than the last. | [V] |
| 47 | Alshomary, M., Ri, N., Apidianaki, M., Muresan, S., McKeown, K. (2024). *Latent space interpretation for stylistic analysis and explainable authorship attribution.* arXiv:2409.07072 | [R] |
| 48 | *Enhancing Representation Generalization in Authorship Identification.* arXiv:2310.00436 | [V] |

## 4. LLM-era authorship analysis

| # | Entry | Status |
|---|---|---|
| 49 | Huang, B. et al. (2024). *Can Large Language Models Identify Authorship?* Findings of EMNLP 2024. Code: github.com/baixianghuang/authorship-llm — Zero/few-shot AA with LLMs, no fine-tuning, plus explicit linguistic features to aid reasoning. | [V] |
| 50 | *Authorship Attribution in the Era of LLMs: Problems, Methodologies, and Challenges.* arXiv:2408.08946 — Good scaffolding for a related-work section that has to cover both human and machine authorship. | [V] |
| 51 | Huang, W., Murakami, A., & Grieve, J. (2024). *ALMs: Authorial Language Models for Authorship Attribution.* arXiv:2401.12005 — Attribution by per-author fine-tuned LM perplexity. 83.6% macro-avg on Blogs50. Includes text-ablation results on how many tokens are needed (40 tokens on Blogs50, 400 on CCAT50 for 70% accuracy) — a method you could port directly to Malayalam. | [V] |
| 52 | Huang, W., Murakami, A., & Grieve, J. (2025). *Attributing authorship via the perplexity of authorial language models.* PLOS ONE. doi:10.1371/journal.pone.0327081 — The journal version of 51. | [V] |
| 53 | *Large Language Models and Forensic Linguistics: Navigating Opportunities and Threats in the Age of Generative AI.* arXiv:2512.06922 | [V] |
| 54 | *Stylometry recognizes human and LLM-generated texts in short samples.* arXiv:2507.00838 | [V] |
| 55 | Rivera-Soto, R. A., Koch, K., Khan, A., Chen, B. Y., Bishop, M., & Andrews, N. (2024). *Few-shot detection of machine-generated text using style representations.* ICLR 2024. | [R] |
| 56 | *Authorship Attribution for LLM-Generated Forged Novels.* EACL 2024 SRW. aclanthology.org/2024.eacl-srw.26 | [V] |

---

# INDIC LANGUAGES

The core of the review. Ordered roughly by depth of existing literature.

## 5. Bengali / Bangla — by far the most developed Indic AA literature

If you want a model for what a mature single-language AA research programme
looks like, this is it: a fifteen-year arc from hand-crafted stylometry through
CNNs and ULMFiT to transformer benchmarks, with two named public datasets. Your
Malayalam work can plausibly be framed as "doing for Malayalam what BAAD16 and
AABL did for Bangla."

| # | Entry | Status |
|---|---|---|
| 57 | Chanda, S., Franke, K., Pal, U., & Wakabayashi, T. (2010). *Text independent writer identification for Bengali script.* ICPR 2010, 2005–2008. — Handwriting, not text; the earliest Indic entry in this lineage. | [R] |
| 58 | Das, S., & Mitra, P. (2011). *Author identification in Bengali literary works.* PReMI 2011. | [R] |
| 59 | Chakraborty, T., & Bandyopadhyay, S. (2011). *Inference of Fine-grained Attributes of Bengali Corpus for Stylometry Detection.* arXiv:1210.3729 — Vocabulary-richness baseline on Tagore. | [V] |
| 60 | Chakraborty, T. (2012). *Authorship Identification in Bengali Literature: a Comparative Analysis.* COLING 2012. arXiv:1208.6268 — 150 stories each from Tagore, Sarat Chandra, and a mixed third class. | [V] |
| 61 | Phani, S., Lahiri, S., & Biswas, A. (2015). *Authorship Attribution in Bengali Language.* ICON 2015, 100–105. aclanthology.org/W15-5915 — 3,000 passages, three authors, character n-grams; feature ranking and learning curves. Held as SOTA for years. | [V] |
| 62 | Rakshit, G., Ghosh, A., Bhattacharyya, P., & Haffari, G. (2015). *Automated analysis of Bangla poetry for classification and poet identification.* ICON 2015, 247–253. | [R] |
| 63 | Phani, S., Lahiri, S., & Biswas, A. (2016). *A machine learning approach for authorship attribution for Bengali blogs.* IALP 2016, 271–274. doi:10.1109/IALP.2016.7875984 | [R] |
| 64 | Hossain, M. T. et al. (2017). *A stylometric analysis on Bengali literature for authorship attribution.* ICCIT 2017. IEEE. | [R] |
| 65 | Nipu, A. S., Pal, U., & Ismail, S. (2017). *A machine learning approach for stylometric analysis of Bangla literature.* ICCIT 2017, 1–5. IEEE. | [R] |
| 66 | Islam, Md. A. et al. (2018). *Authorship attribution on Bengali literature using stylometric features.* | [R] |
| 67 | Chowdhury, H. A. et al. (2018). *Authorship attribution in Bengali using word embedding models.* | [R] |
| 68 | Chowdhury, H. A., Imon, Md. A. H., & Islam, Md. S. (2020). *Authorship Attribution in Bangla Literature using Character-level CNN.* arXiv:2001.05316 — Custom crawl, 13.4M+ words, 750-word documents; character-level fastText embeddings reach ~98% on six authors. | [V] |
| 69 | Ibn Ahmad, S., Alam, L., & Hoque, M. M. (2020). *An empirical framework to identify authorship from Bengali literary works.* ICONCS 2020, LNICST 325, 465–476. doi:10.1007/978-3-030-52856-0_37 | [R] |
| 70 | *A Stylometric Approach for Author Attribution System Using Neural Network and Machine Learning Classifiers.* ICCA 2020. doi:10.1145/3377049.3377079 — Eight Bangla political columnists; MLP + SVM with voting ensembles. | [V] |
| 71 | Khatun, A. (2020). *BAAD16: Bangla Authorship Attribution Dataset.* Mendeley Data. doi:10.17632/6D9JRKGTVV.4 — 16 authors. The de facto Bangla benchmark. | [V] |
| 72 | Khatun, A., Rahman, A., Islam, Md. S., Chowdhury, H. A., & Tasnim, A. (2022). *Authorship Attribution in Bangla Literature (AABL) via Transfer Learning using ULMFiT.* ACM TALLIP. doi:10.1145/3530691; arXiv:2403.05519 — Explicitly targets scalability in number of authors and low samples per author. Read this one closely: those are exactly your constraints. | [V] |
| 73 | *Deep Bangla Authorship Attribution Using Transformer Models.* Springer LNCS (2021). doi:10.1007/978-3-030-91434-9_11 | [V] |
| 74 | *The Word2vec Graph Model for Author Attribution and Genre Detection in Literary Analysis.* arXiv:2310.16972 — Bengali literature; jointly does author attribution and genre detection, with per-genre breakdowns showing m-BERT collapsing on short stories where TF-IDF and character n-grams hold up. Directly relevant to your fiction/non-fiction split. | [V] |
| 75 | **Moosa, A. M., Sultana, N., Moosa, M. M., & Hossain, Md. M. (2025). *BARD10: A New Benchmark Reveals Significance of Bangla Stop-Words in Authorship Attribution.* arXiv:2511.08085 (11 Nov 2025)** — The most recent Indic AA paper I found. New 10-author blog/opinion corpus; compares SVM, Bangla BERT, XGBoost, MLP under uniform preprocessing on BARD10 and BAAD16. TF-IDF + linear SVM wins both (macro-F1 0.997 on BAAD16, 0.921 on BARD10); Bangla BERT trails by up to five points, attributed to 512-token truncation and to stop-words carrying real authorial signal. Token-level Δ-Recall analysis. **Read this before you design your Malayalam experiments — it is the strongest recent argument against assuming a transformer baseline will win.** | [V] |
| 76 | Hossain, Md. R., Hoque, M. M., Dewan, M. A. A., Hoque, E., & Siddique, N. (2025). *AuthorNet: Leveraging attention-based early fusion of transformers for low-resource authorship attribution.* Expert Systems with Applications 262, 125643. | [R] |
| 77 | Abdullah, M., Kaisar, M., Islam, A., & Mojumder, M. (2025). *A Hybrid Deep Learning and Stylometric Feature-Based Framework for Authorship Attribution in Bangla Literature.* | [R] |
| 78 | Tasnim, R., Chowdhury, M., & Rahman, Md. A. (2024). *BN-AuthProf: Benchmarking Machine Learning for Bangla Author Profiling on Social Media Texts.* arXiv:2412.02058 — Profiling (age/gender), not attribution, but a useful adjacent dataset paper. | [V] |
| 79 | *Emotion Analysis of Social Media Bangla Text and Its Impact on Identifying the Author's Gender.* arXiv:2411.04524 | [V] |
| 80 | *Author attribution from the lyrics of Bengali songs.* (machine-learning approach, datasets D2A/D4A/D7A) | [R] |

## 6. Urdu — the best-resourced *corpus* work in South Asia

Urdu is where someone actually solved the corpus problem properly. If you want a
template for arguing "existing Indic AA datasets are too small to be
real-world," these two papers made that argument and then fixed it.

| # | Entry | Status |
|---|---|---|
| 81 | *Authorship Attribution for a Resource Poor Language — Urdu.* ACM TALLIP (2021). doi:10.1145/3487061 — Argues existing Urdu AA studies used under 20 candidate authors, far from real-world. Builds a corpus of 21,938 news articles by 94 authors, 2.6M+ tokens. Synthesises hundreds of stylometry features from the literature down to 194 applicable to Urdu, in five categories (character, word, sentence, paragraph, document). 66 experiments over four traditional and three deep-learning techniques. **The feature taxonomy alone is worth adapting to Malayalam.** | [V] |
| 82 | *UrduAI: Writeprints for Urdu Authorship Identification.* ACM TALLIP (2021). doi:10.1145/3476467 — Writeprints adapted to Urdu; 6× more candidate authors than prior work; handles open-set attribution (rejecting texts by authors outside the candidate set). Benchmarks against FAUT-W15 and LIP-W12. | [V] |
| 83 | Anwar, W., Bajwa, I. S., Choudhary, M. A., & Ramzan, S. *An Empirical Study on Forensic Analysis of Urdu Text Using LDA-Based Authorship Attribution.* Scientific Programming / IEEE Access. — Instance-based vs. profile-based LDA; unsupervised. | [R] |
| 84 | *Authorship Attribution in Urdu Poetry.* (2020) — Corpus scraped from Urdu Library, Rekhta, Urdu Web, Sukhansara, Iqbal. | [V] |
| 85 | Siddiqui, ... & Rubab, ... *Poet Attribution of Urdu Ghazals using Deep Learning.* — First transformer-based poet attribution for Urdu ghazals. | [V] |

## 7. Other Indic languages

### 7.1 Hindi

| # | Entry | Status |
|---|---|---|
| 86 | Srijan, S. (2014). *Classification of Hindi authors on the basis of author writing style.* CS365 project report, IIT Kanpur. cse.iitk.ac.in/users/cs365/2014/_submissions/srijans/project/report.pdf — The Hindisamay corpus; SVM and k-means over uni/bi/trigrams and MDA. **This is the source of your legacy corpus.** Note in your paper that it is an undergraduate course project, not peer-reviewed — that is a defensible reason for building a new Malayalam resource. | [V] |
| 87 | *Author Identification of Hindi Poetry.* IJSTR (2020). ijstr.org/final-print/mar2020/Author-Identification-Of-Hindi-Poetry.pdf — 100 poems each from three authors; J48 for feature selection, then SMO / Bayes Net / Naïve Bayes. | [V] |
| 88 | *Author Identification of Hindi Stories.* International Journal of Psychosocial Rehabilitation (2020). — Same team and method, 70 stories per author. | [V] |
| 89 | *Authorship Attribution in Hindi Literary Texts: An Exploration of Traditional Linguistic Approaches and Experimentation with Multilingual BERT.* Research Square (2024). doi via researchsquare.com/article/rs-5462231/v1 — Curated Hindi short-story dataset; **finds traditional n-gram / BoW / TF-IDF methods outperform m-BERT**. Second independent Indic data point for the pattern in entry 75. | [V] |
| 90 | *Stylometric Analysis of Genre in Hindi Literature.* (2024) — Genre rather than author, but same feature machinery; discusses Agyeya and Jainendra Kumar misgrouping cases. | [V] |
| 91 | Sharma, A., Nandan, A., & Ralhan, R. (2018). *An Investigation of Supervised Learning Methods for Authorship Attribution in Short Hinglish Texts using Char & Word N-grams.* ACM TALLIP; arXiv:1812.10281 — Code-mixed / romanised. | [V] |

### 7.2 Telugu

| # | Entry | Status |
|---|---|---|
| 92 | Prasad, S. N., Narsimha, V. B., Reddy, P. V., & Babu, A. V. (2015). *Influence of lexical, syntactic and structural features and their combination on authorship attribution for Telugu text.* Procedia Computer Science 48(C), 58–64. — Uses the Telugu Morphological Analyser for stemming; explicitly frames Dravidian morphological complexity as the core difficulty. **Closest methodological precedent for Malayalam of anything in this list.** | [V] |
| 93 | *Authorship Attribution of Telugu Texts Based on Shallow Features.* JATIT 85(1). jatit.org/volumes/Vol85No1/13Vol85No1.pdf — Function-word frequencies and POS on editorial articles by different journalists; deliberately excludes lexical features because genre is held constant. | [V] |
| 94 | *Authorship Attribution based on Data Compression for Telugu Text.* — Compression-distance approach; a strong, feature-free baseline worth including. | [V] |
| 95 | Pandian, A., Ramalingam, V. V., Manikandan, K., Jeevan, V., & Krishna, P. S. (2018). *Author identification for Telugu classical poems.* International Journal of Engineering & Technology 7(4.19), 26–31. — 836 Telugu poems; 88.69% with J48. | [V] |
| 96 | Pandian, A., Manikandan, K., Ramalingam, V. V., & Reddy, V. J. (2018). *Comparative studies of author identification algorithms for Telugu classical poems.* | [V] |

### 7.3 Kannada

| # | Entry | Status |
|---|---|---|
| 97 | Chandrika, C. P., & Kallimani, J. S. (2022). *Authorship attribution for Kannada text using profile based approach.* LNNS 237, 679–688. doi:10.1007/978-981-16-6407-6_58 — States that Kannada digital-text AA had not been attempted before. | [V] |
| 98 | *Instance Based Authorship Attribution for Kannada Text Using Amalgamation of Character and Word N-grams Technique.* Springer (2022). doi:10.1007/978-981-19-2281-7_51 | [V] |
| 99 | *Authorship Attribution on Kannada Text using Bi-LSTM.* IJACSA 13(9). thesai.org/Downloads/Volume13No9/Paper_111-Authorship_Attribution_on_Kannada_Text.pdf — POS + n-gram features into a BiLSTM. | [V] |

### 7.4 Tamil

| # | Entry | Status |
|---|---|---|
| 100 | Pandian, A., Ramalingam, V. V., Manikandan, K., & Vishnu Preet, R. P. (2016). *Authorship Identification for Tamil Classical Poem (Mukkoodar Pallu) using C4.5 Algorithm.* Indian J. Science and Technology 9(47). | [V] |
| 101 | Pandian, A. et al. (2017). *Authorship Identification for Tamil Classical Poem (Mukkoodar Pallu) using Bayes Net Algorithm.* Indian J. Science and Technology. — 94.1% accuracy, up from 88.23%. | [V] |
| 102 | Pandian, A., Ramalingam, V. V., Manikandan, K., & Vishnu Preet, R. P. (2018). *Authorship identification for Tamil classical poem using subspace discriminant algorithm.* J. Physics: Conf. Series 1000(1), 1–9. | [R] |
| 103 | *Poetic and Semantic Features for Lyricist Identification from Tamil Film Lyrics.* SN Computer Science (2022). doi:10.1007/s42979-022-01416-2 — **The largest-scale Indic AA study I found: 15,286 lyric documents across 113 lyricists.** Statistical + linguistic + poetic + semantic features; SVM best. | [V] |
| 104 | Chandrasekaran, R., & Manimannan, G. (2013). *Use of generalized regression neural network in authorship attribution.* International Journal of Computer Applications 62(4), 7–10. — Disputed-authorship articles by Bharathiar, Subramaniya Iyer, and T. V. Kalyanasundaram in the pre-independence magazine *India*. A genuine historical attribution problem, not a synthetic one. | [R] |
| 105 | *Authorship Attribution to Tamil language email using lexical and syntactic signatures with FLD, RBF and echo state neural networks.* | [R] |
| 106 | *Tamil Lyrics Corpus: Analysis and Experiments.* | [V] |

### 7.5 Marathi

| # | Entry | Status |
|---|---|---|
| 107 | Kale, S. D., & Prasad, R. (2018). *Author Identification using Sequential Minimal Optimization with rule-based Decision Tree on Indian Literature in Marathi.* Procedia Computer Science. doi:10.1016/j.procs.2018.05.023 — Notes explicitly that no prior AA experiment existed for Marathi. | [V] |
| 108 | *Influence of Language-Specific Features for Author Identification on Indian Literature in Marathi.* Springer (2020). doi:10.1007/978-981-15-2475-2_59 — Argues language-specific lexical features beat generic ones. Its reference list is one of the best single sources of Indic AA citations available. | [V] |
| 109 | *Author Identification for Marathi Language.* ASTESJ 5(2). astesj.com/v05/i02/p56/ — Frames AA as intrinsic plagiarism detection. | [V] |

### 7.6 Assamese, Punjabi, Gujarati

| # | Entry | Status |
|---|---|---|
| 110 | *Authorship Attribution for Assamese Language Documents: Initial Results.* Springer LNNS (2023). doi:10.1007/978-3-031-47224-4_21 — Manually collected and annotated Assamese literary corpus; claims to be the first AA attempt in Assamese. **Read this one for its framing: it is the closest structural analogue to your project — a first-of-language literary AA paper built on a hand-collected corpus.** | [V] |
| 111 | Kaur, N., & Verma, A. (2015). *Authorship attribution of Punjabi poetry using SVM classifier.* IJARCSSE 5(5), 1055–1061. | [R] |
| 112 | *Optimum Parameter Selection for K.L.D. Based Authorship Attribution in Gujarati.* — KL-divergence author profiles; compares against Z-score. | [R] |

### 7.7 Malayalam

The entire dedicated literature, such as it is.

| # | Entry | Status |
|---|---|---|
| 113 | Elayidom, M. S., Jose, C., Puthussery, A., & Sasi, N. K. (2013). *Text Classification For Authorship Attribution Analysis.* arXiv:1310.4909 — From CUSAT and MG University, Kerala. Fuzzy classifier vs. SVM, then combined; SVM alone beats fuzzy, combination beats both. **But the work itself is not on Malayalam text** — it is a general stylometry pipeline from a Kerala group. Cite it for regional provenance, not as Malayalam AA. | [V] |
| 114 | Kim, Zhang & Jurgens (2025) — see entry 30. Malayalam as one of 22 evaluated languages, Wikipedia domain, 1,718 authors. **The only quantitative Malayalam AA result in the literature.** | [V] |
| 115 | *Shared Task on Indian Native Language Identification (INLI) @ FIRE.* nlp.amrita.edu/INLI/ — Malayalam is one of six languages, but the task is identifying an author's L1 from their *English* Facebook comments. Adjacent, not AA. | [V] |

That is the complete set. Everything else returned by searches on "Malayalam
authorship" was code-mixed sentiment/offensive-language work, speech and TTS
corpora (IMaSC), morphological analysis, or palm-leaf manuscript OCR — different
tasks entirely.

### 7.8 Sanskrit — the authorship *verification* tradition

Methodologically distinct and worth a subsection in your review: this community
does open-set verification on disputed attributions, not closed-set
classification, and it is the only Indic strand engaging seriously with the
General Imposters framework.

| # | Entry | Status |
|---|---|---|
| 116 | Gussner, R. E. (1976). *A Stylometric Study of the Authorship of Seventeen Sanskrit Hymns Attributed to Śaṅkara.* JAOS 96(2), 259–267. — Predates most computational AA. | [R] |
| 117 | Andrijanić, I., & Bąkowski, J. *The authorship of the Chāndogyopaniṣad-Bhāṣya: A stylometric approach.* — Compares manual sandhi resolution against automatic 3-gram segmentation; word segmentation wins on texts of confirmed authorship. | [V] |
| 118 | Andrijanić, I., & Bąkowski, J. *Stylometry Then and Now: Authorship Verification of Vivekacūḍāmaṇi and Vedāntic Stotras.* — General Imposters framework; reports up to 15% error from machine word-separation. | [V] |
| 119 | Andrijanić, I., & Bąkowski, J. *On the Authenticity of Prose Writings Attributed to Śaṅkara.* — 18 prose commentaries; GI success rate 77.5–80%; character trigrams + Cosine Delta best. | [V] |
| 120 | *Śaṅkara and the Authorship of the Īśopaniṣadbhāṣya and the Kaṭhopaniṣadbhāṣya.* International Journal of Hindu Studies (2020). doi:10.1007/s11407-020-09279-z | [V] |
| 121 | *The Authorship of the Sanatsujātīya- and Viṣṇusahasranāma-Bhāṣya Attributed to Śaṅkara.* IJHS (2025). doi:10.1007/s11407-025-09405-9 — Philological rebuttal to the stylometric verdict. Useful if you want to say something about interpretability and expert trust. | [V] |
| 122 | *Computational Sanskrit and Digital Humanities* workshop series (ACL) — lists stylometry and authorship attribution as a standing topic. | [V] |

### 7.9 Sinhala and other South Asian

| # | Entry | Status |
|---|---|---|
| 123 | *Stylomech: Unveiling Authorship via Computational Stylometry in English and Romanized Sinhala.* arXiv:2501.09561 — Trains on similarity scores between text pairs rather than raw text, to sidestep data scarcity. A design worth stealing if your per-author Malayalam counts stay low. | [V] |

## 8. Shared tasks, benchmarks and evaluation infrastructure

| # | Entry | Status |
|---|---|---|
| 124 | PAN @ CLEF author identification / verification overviews, 2013–2025 (Juola & Stamatatos 2013; Stamatatos et al. 2014, 2015; Kestemont et al. 2018–2021). — The standard evaluation framework; entry 30 uses PAN 2013–2015 verification sets for zero-shot evaluation. | [V] |
| 125 | *Shared Tasks on Authorship Analysis at PAN 2020.* | [V] |
| 126 | *Multilingual Author Profiling on SMS Track (MAPonSMS) at FIRE 2018.* — Age and gender from SMS across India, Pakistan, Nepal, Bangladesh. | [V] |
| 127 | FIRE evaluation initiative reports (Mehta et al., 2020–2021) — includes an authorship identification track for source code. Confirms FIRE has hosted authorship work but never a text AA track for Indian languages. | [V] |
| 128 | *Author Profiling in Code-Mixed WhatsApp Messages Using Stacked Convolution Networks and Contextualized Embedding Based Text Augmentation* (Tamil). | [R] |
| 129 | *BhashaSutra: A Task-Centric Unified Survey of Indian NLP Datasets, Corpora, and Resources.* arXiv:2604.18423 — Use this to check whether any Malayalam author-labelled resource exists that I missed. | [V] |
| 130 | *Bhaasha, Bhasa, Zaban: A Survey for Low-Resourced Languages in South Asia.* arXiv:2509.11570 | [V] |
| 131 | Kakwani, D. et al. (2020). *IndicNLPSuite / IndicBERT / IndicCorp / IndicGLUE.* — Malayalam is covered. Your most likely pretrained encoders: IndicBERT, MuRIL, XLM-R. | [V] |
| 132 | *IndicXTREME: A Benchmark for Extreme Multilingual Evaluation of Indian Languages.* Findings of ACL 2023. | [R] |

## 9. Low-resource AA in non-Indic languages — methodological templates

These are the papers you cite when a reviewer asks "why is this hard, and what
have others done under the same constraints?"

| # | Entry | Status |
|---|---|---|
| 133 | Nitu, M., & Dascalu, M. (2024). *Authorship Attribution in Less-Resourced Languages: A Hybrid Transformer Approach for Romanian.* Applied Sciences 14(7), 2700. doi:10.3390/app14072700 — Concatenates handcrafted linguistic features with Romanian BERT embeddings; evaluates on both full texts and paragraphs, 19 authors and a 10-author subset. **The closest architectural template for what a hybrid Malayalam system could look like.** | [V] |
| 134 | Avram, S.-M. (2023). *BERT-based authorship attribution on the Romanian dataset called ROST.* arXiv:2301.12500 | [R] |
| 135 | Gabrovšek, G., Peer, P., Emeršič, Ž., & Batagelj, B. (2023). *Authorship attribution on short texts in the Slovenian language.* Applied Sciences 13(19). | [R] |
| 136 | Misini, A., Canhasi, E., Kadriu, A., & Fetahi, E. (2024). *Automatic authorship attribution in Albanian texts.* PLOS ONE 19(10). | [R] |
| 137 | De Langhe, L., De Clercq, O., & Hoste, V. (2024). *Unsupervised authorship attribution for medieval Latin using transformer-based embeddings.* LT4HALA @ LREC-COLING 2024, 57–64. | [R] |
| 138 | *PARSI: Persian Authorship Recognition via Stylometric Integration.* arXiv:2506.21840 — 67 poets; multi-input neural framework. Poetry-specific features (metre, rhyme) that would transfer to Malayalam verse. | [V] |
| 139 | *Integrated ensemble of BERT- and feature-based models for authorship attribution in Japanese literary works.* arXiv:2504.08527 / PMC12497813 — Another non-Latin-script literary AA ensemble. | [V] |
| 140 | Saygılı, N. S., Amghar, T., Levrat, B., & Acarman, T. *Taking advantage of Turkish characteristic features to achieve authorship attribution problems for Turkish.* — Agglutinative-language AA; morphological features. Relevant precedent for a language like Malayalam. | [R] |
| 141 | Ramezani, R. (2021). *A language-independent authorship attribution approach for author identification of text documents.* Expert Systems with Applications 180, 115139. | [R] |

## 10. Quantum / QNLP intersection

Searched specifically for quantum or hybrid quantum-classical methods applied to
authorship attribution, verification, or stylometry. **I found nothing.** The
nearest work applies hybrid quantum-classical models to adjacent text-forensics
tasks:

| # | Entry | Status |
|---|---|---|
| 142 | *HQML-NLP: A hybrid quantum machine learning framework for scholarly AI-text detection.* Applied Soft Computing (2026). doi:10.1016/j.asoc.2026.[see ScienceDirect S1568494626000827] — SBERT 384-d embeddings + six-qubit two-layer PQC → 390-d hybrid feature space. Claims accuracy competitive with transformer and ensemble detectors at >2000× fewer trainable parameters. Frames itself as AI-assisted **authorship verification**. This is the closest published thing to what you are proposing. | [V] |
| 143 | *Beyond Classical AI: Detecting Fake News with Hybrid Quantum Neural Networks.* Applied Sciences 15(15), 8300 (2025). doi:10.3390/app15158300 — HQDNN in PyTorch + PennyLane on LIAR; benchmarked against SetFit. Higher recall, lower precision. | [V] |

**Read this as opportunity, not as absence of evidence.** The parameter-efficiency
argument in entry 142 is the strongest available justification for a quantum
component on a small corpus: if your Malayalam corpus caps out at a few hundred
documents per author, a PQC head over frozen classical embeddings is a
defensible design, not a gimmick. That framing — *quantum as a low-data,
low-parameter classifier head, not as a speedup claim* — is what will survive
review.

---

## 11. Gap analysis

Five things the literature actually establishes, and what each implies.

**1. Malayalam literary AA is genuinely unoccupied.**
Bengali has fifteen years and two named benchmarks. Urdu has 94-author,
2.6M-token corpora. Telugu, Kannada, Tamil, Marathi, Assamese, Punjabi and
Gujarati each have at least one dedicated study. Malayalam has one number, in
one table, of one multilingual paper, on Wikipedia text. This is a first-of-
language contribution and you should claim it as one — the Assamese paper
(entry 110) shows exactly how that claim is framed and published.

**2. Transformers do not automatically win on Indic literary AA.**
Three independent findings: BARD10 (entry 75) has TF-IDF + linear SVM beating
Bangla BERT on both Bangla benchmarks; the Hindi m-BERT study (entry 89) finds
traditional methods beat m-BERT on Hindi short stories; the word2vec-graph paper
(entry 74) finds m-BERT collapsing on short Bengali stories where TF-IDF and
character n-grams hold. The stated causes are 512-token truncation on long
literary documents and the destruction of high-frequency function-word signal by
subword tokenisation. **Your experiment design must include a strong sparse
baseline, and you should expect it to be competitive.** A paper that reports only
a transformer result will be weaker, not stronger.

**3. Whole-work documents create a length problem nobody in Indic AA has solved.**
You have decided to keep each Sayahna work as one document. That collides
directly with the 512-token limit. The literature offers three routes:
hierarchical or long-context encoders; the LUAR-style approach of sampling many
short excerpts and pooling (entry 31); or per-author language-model perplexity
over the whole document (entries 51–52), which has no length ceiling at all.
None of these has been tried on an Indic literary corpus. That is a
methodological contribution available to you independent of the quantum angle.

**4. Your cross-genre problem is a known, named, publishable problem.**
Your merged corpus is ~673 works, 24 authors, but only one author has 4+ works
in both fiction and non-fiction — so cross-genre attribution is off the table as
designed. Entries 18–20, 26–27 and 45 are the literature on exactly this. Two
honest options: report single-genre attribution and cite the cross-genre
literature as a stated limitation and future work; or deliberately extend the
corpus to recover a cross-genre subset, which would make the resource
substantially more valuable than a single-genre one. The second is more work and
a much better paper.

**5. Malayalam morphology is the language-specific angle.**
The Telugu work (entry 92) built its pipeline around a morphological analyser
precisely because Dravidian agglutination breaks word-level features. Sapkota et
al. (entry 17) show affix n-grams carry most of the character n-gram signal in
English — where affixation is comparatively thin. In an agglutinative
Malayalam this should be *more* true, not less, and nobody has tested it. A
controlled study of morpheme-aware versus surface-form features for Malayalam AA
is a clean, self-contained result.

---

## 12. Reading checklist

Ordered by what unblocks the most downstream work. No dates attached — these are
deliverables, and each one is done when the stated artefact exists.

### Tier 1 — read before designing any experiment

| Read | Entry | Done when |
|---|---|---|
| ☐ | 30 — Kim, Zhang & Jurgens 2025, multilingual AR | You have the Malayalam row from Table A5/A18 in your notes and have downloaded the HF model |
| ☐ | 75 — BARD10 | You can state in one paragraph why TF-IDF+SVM beat Bangla BERT, and whether the cause applies to Malayalam |
| ☐ | 72 — AABL / ULMFiT | You have their scalability protocol (varying author count, samples per author) written down as a design you can replicate |
| ☐ | 81 — Urdu TALLIP corpus paper | The 194-feature taxonomy is extracted into a spreadsheet with a "applies to Malayalam?" column |
| ☐ | 110 — Assamese initial results | You have a one-page outline of how a first-of-language AA paper is structured |
| ☐ | 8 — Tyo et al., State of the Art | Your planned evaluation protocol has been checked against their critique |

**Tier 1 deliverable:** a two-page experiment-design memo — baselines, splits,
metrics, author/document counts — that you can put in front of your guide.

### Tier 2 — read while building the corpus and baselines

| Read | Entry | Done when |
|---|---|---|
| ☐ | 92 — Telugu morphological features | You have decided whether to use a Malayalam morphological analyser and named which one |
| ☐ | 17, 19, 20 — character n-grams, distortion, preprocessing | Your feature extractor implements typed character n-grams |
| ☐ | 26 — Topic Confusion Task | You have a concrete plan for proving your model is not learning topic |
| ☐ | 1, 6 — Stamatatos 2009 + Xie et al. 2024 | Related-work section drafted |
| ☐ | 31, 51 — LUAR + ALMs | You have picked your long-document strategy |
| ☐ | 133 — Romanian hybrid transformer | Your hybrid architecture is sketched |

**Tier 2 deliverable:** a reproducible baseline table on your Malayalam corpus —
TF-IDF+SVM, character n-grams, IndicBERT/MuRIL, and the entry-30 multilingual AR
model zero-shot — with book-disjoint splits.

### Tier 3 — read when writing the quantum section

| Read | Entry | Done when |
|---|---|---|
| ☐ | 142 — HQML-NLP | You can state the parameter-count argument in two sentences |
| ☐ | 143 — HQDNN fake news | You have PennyLane + PyTorch running end-to-end on a toy classifier |
| ☐ | 117–119 — Sanskrit General Imposters | You have decided whether verification (open-set) is a better fit for the quantum component than closed-set attribution |

**Tier 3 deliverable:** a one-page justification for the quantum component that
does not rely on a speedup claim.

### Verification pass

| Task | Done when |
|---|---|
| ☐ Pull every `[R]` entry you intend to cite | Each has a confirmed DOI or PDF, or is dropped |
| ☐ Cross-check entry 108's reference list | Any Indic AA paper it cites that is not in this file has been added |
| ☐ Cross-check entry 129 (BhashaSutra) for Malayalam author-labelled resources | Confirmed that none exists, or found one |
| ☐ Re-run the Malayalam search in three months | New arXiv preprints checked; this area is moving |

---

## 13. Known gaps in this review itself

Being straight about what this compilation does not cover:

- **Google Scholar and Semantic Scholar were not queried directly.** Everything
  here came through general web search. Regional Indian journals and conference
  proceedings are poorly indexed by web search; a Scholar citation-chase from
  entries 108, 110 and 6 will surface more, especially for Kannada, Gujarati and
  Punjabi.
- **Non-English-language publications are almost certainly missed** — there may
  be Malayalam-language computational-linguistics work in Kerala university
  theses or ICFOSS/Amrita outputs that never reached an indexed venue. Worth
  asking your guide directly.
- **Theses and dissertations are not covered.** Shodhganga is worth a search for
  Malayalam stylometry.
- **`[R]` entries are unverified.** Roughly a third of the list.
- **Handwriting-based writer identification was deliberately excluded** except
  entry 57. Different task, different literature.
