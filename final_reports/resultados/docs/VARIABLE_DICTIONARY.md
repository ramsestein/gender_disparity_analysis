# Variable Dictionary and Technical Glossary

## Study of Gender Disparity in Scientific Debates

This document provides complete definitions of all extracted and calculated variables, as well as a glossary of technical terms employed in the methodology and statistical analysis.

---

# PART I: DATASET VARIABLES

---

## 1. Identifiers and Metadata

### `intervention_id`
- **Type:** Integer
- **Range:** 1 to N (sequential per session)
- **Definition:** Unique identifier that numbers each intervention sequentially within a session. Enables traceability and unequivocal reference of each unit of analysis.

### `session`
- **Type:** String
- **Definition:** Unique identifier of the session or debate from which the intervention originates. Corresponds to the name of the original video/audio file processed.

### `speaker`
- **Type:** String
- **Format:** SPEAKER_XX (e.g. SPEAKER_01, SPEAKER_02)
- **Definition:** Anonymous identifier automatically assigned by the diarization algorithm to each unique speaker detected in the session.

### `gender`
- **Type:** Categorical
- **Values:** `male`, `female`
- **Definition:** Speaker's gender, classified by the dual validation system (pitch analysis + deep learning model) with human arbitration in case of discrepancy.

### `gender_bin`
- **Type:** Binary
- **Values:** 1 (male), 0 (female)
- **Definition:** Numerical encoding of gender for use in regression models and statistical analysis.

### `turn_number`
- **Type:** Integer
- **Range:** 1 to N
- **Definition:** Position of the intervention in the logical sequence of speaking turns. Increments each time there is a speaker change, regardless of overlaps.

---

## 2. Temporal Variables

### `start_time`
- **Type:** Float
- **Unit:** Seconds
- **Definition:** Timestamp of the intervention start, measured from the beginning of the session audio.

### `end_time`
- **Type:** Float
- **Unit:** Seconds
- **Definition:** Timestamp of the intervention end, measured from the beginning of the session audio.

### `duration`
- **Type:** Float
- **Unit:** Seconds
- **Range:** >0.3 (shorter interventions were filtered)
- **Definition:** Net duration of the intervention, calculated as `end_time - start_time`. Represents the total time the speaker held the floor.

### `latency_s`
- **Type:** Float
- **Unit:** Seconds
- **Range:** Can be negative
- **Definition:** Silence (or overlap) time between the end of the previous speaker and the start of the current speaker. Negative values indicate the speaker began speaking before the previous one finished (overlap/interruption).

### `session_phase`
- **Type:** Float
- **Range:** 0.0 to 1.0
- **Definition:** Relative position of the intervention within the total session duration. A value of 0.0 indicates the start of the session; 1.0 indicates the end. Enables temporal evolution analysis.

### `phase_quartile`
- **Type:** Categorical
- **Values:** Q1, Q2, Q3, Q4
- **Definition:** Session quartile in which the intervention occurs:
  - **Q1:** First 25% of the session (opening)
  - **Q2:** 25-50% of the session
  - **Q3:** 50-75% of the session
  - **Q4:** Last 25% of the session (closing)

---

## 3. Turn Dynamics and Interruption

### `has_overlap`
- **Type:** Boolean
- **Values:** True/False (or 1/0)
- **Definition:** Indicates whether at any point during the intervention there was simultaneous speech with another speaker. Detected when temporal ranges of two interventions overlap.

### `overlap_duration`
- **Type:** Float
- **Unit:** Seconds
- **Range:** ≥0
- **Definition:** Total duration of time the intervention overlapped with another speaker's speech. If there was no overlap, the value is 0.

### `interrupts_previous`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Indicates whether the current speaker began speaking before the previous speaker finished their intervention. Captures the "intrusive entry" into another's turn. Determined when `latency_s < 0`.

### `interrupted_by_next`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Indicates whether the current intervention was cut off by the next speaker (i.e., if the next speaker started before this one finished). It is the complement of `interrupts_previous` from the interrupted speaker's perspective.

### `interruption_success`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Indicates whether a speaker who interrupted the previous one managed to consolidate the speaking turn, i.e., whether they succeeded in silencing the previous speaker and holding the floor. Marked as successful when the interrupted speaker stops speaking after the overlap.

### `is_backchannel`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Identifies short "supportive speech" interventions or verbal feedback that do not intend to take the speaking turn. Examples: "mhm", "yeah", "I see", "okay", "right". Detected by short duration (<2 seconds) and specific lexical patterns.

---

## 4. Basic Linguistic Metrics

### `text`
- **Type:** String
- **Definition:** Literal transcription of the verbal content of the intervention, obtained via the Whisper model.

### `word_count`
- **Type:** Integer
- **Range:** ≥0
- **Definition:** Total number of words (lexical tokens) in the intervention, excluding punctuation marks.

### `wpm` (Words Per Minute)
- **Type:** Float
- **Unit:** Words per minute
- **Typical range:** 100-250
- **Definition:** Speaker's speech rate, calculated as `(word_count / duration) × 60`. Measures fluency and rhythm of speech.

### `lexical_diversity`
- **Type:** Float
- **Range:** 0.0 to 1.0
- **Synonym:** TTR (Type-Token Ratio)
- **Definition:** Proportion of unique words (types) over total words (tokens) in the intervention. A value close to 1 indicates high lexical variety (few repetitions); low values indicate repetitive vocabulary. Calculated on lemmas to normalize morphological variations.

---

## 5. Grammatical and Syntactic Variables

### `is_question`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Indicates whether the intervention constitutes a question. Detected through:
  - Presence of final question mark
  - Interrogative words (what, who, when, where, why, how) in initial position
  - Interrogative syntactic structure (subject-verb inversion)

### `num_imperatives`
- **Type:** Integer
- **Range:** ≥0
- **Definition:** Number of verbs in imperative mood detected in the intervention. Imperatives represent orders, direct instructions, or exhortations, and are indicators of a directive communicative style.

---

## 6. Pragmatic Markers

### `has_hedge`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Indicates the use of mitigating or tentative language that reduces the assertive force of the utterance. Hedges convey uncertainty, tentativeness, or negative politeness.
- **Detected examples:** "I think", "maybe", "perhaps", "probably", "possibly", "sort of", "kind of", "seems", "appears", "actually", "just", "a bit", "somewhat"

### `has_apology`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Indicates the presence of explicit apology expressions.
- **Detected examples:** "sorry", "I apologize", "excuse me", "pardon", "forgive me", "my apologies"

### `has_courtesy`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Indicates the use of positive politeness or deference markers.
- **Detected examples:** "please", "thank you", "thanks", "kindly", "I appreciate", "if you don't mind"

### `has_vulnerability`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Indicates explicit admission of lack of knowledge, uncertainty, or self-limitation.
- **Detected examples:** "I'm not sure", "I don't know", "I'm uncertain", "I'm confused", "I need help", "I struggle with"

### `has_agreement`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Indicates expression of conformity or agreement with the previous speaker.
- **Detected examples:** "I agree", "absolutely", "exactly", "that's right", "good point", "you're right", "yes"

### `has_disagreement`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Indicates expression of discrepancy or disagreement with the previous speaker.
- **Detected examples:** "I disagree", "I don't think", "however", "but", "on the contrary", "not necessarily", "actually...no"

---

## 7. Authority and Academic Credit

### `has_title`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Indicates the use of academic or professional titles when referring to other participants or oneself.
- **Detected examples:** "Doctor", "Dr.", "Professor", "Prof.", "PhD", "colleague", "expert"

### `has_attribution`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Indicates explicit recognition of idea authorship, giving credit to the original source.
- **Detected examples:** "As Dr. X said", "According to", "X mentioned", "X pointed out", "Building on what X said", "As my colleague noted"

---

## 8. Echoing Score and Appropriation

### `echoing_score`
- **Type:** Float
- **Range:** 0.0 to 1.0
- **Definition:** Index of lexical similarity between the current intervention and the immediately preceding one. Measures what proportion of the content vocabulary (nouns, adjectives, main verbs) of the previous speaker is taken up by the current speaker. Calculated as the Jaccard coefficient on lemma sets.
- **Interpretation:**
  - **Low (<0.1):** Little thematic continuity, topic change
  - **Medium (0.1-0.3):** Normal thematic continuity
  - **High (>0.3):** Strong uptake of previous content

### `appropriation`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Indicates whether an intervention presents idea appropriation, operationalized as the combination of:
  - High echoing score (echoing_score > 0.3): the speaker takes up previous speaker's content
  - Absence of attribution (has_attribution = False): does not give credit to the source
- **Interpretation:** Captures situations where a speaker repeats or reformulates another's ideas without acknowledging their origin.

### `ignored`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Applies only to interventions that are questions. Indicates whether the subsequent response showed low lexical echoing (echoing_score < 0.1), suggesting the question was not addressed or was thematically ignored.

---

## 9. Composite Indices

### `conflict_score`
- **Type:** Integer
- **Typical range:** -2 to +5
- **Definition:** Conflict index of the intervention. Weighted sum capturing the level of confrontation:
  - **Adds:** has_disagreement (+1), num_imperatives (+1), interrupts_previous (+1)
  - **Subtracts:** has_agreement (-1), has_courtesy (-1)
- **Interpretation:** High positive values indicate confrontational interventions; negative values indicate conciliatory interventions.

### `peak_conflict`
- **Type:** Boolean
- **Values:** True/False
- **Definition:** Indicates whether the intervention is in the top 90th percentile of conflict in the complete dataset. Allows identification of moments of maximum tension.

### `assertiveness_score`
- **Type:** Integer
- **Typical range:** 0 to 5
- **Definition:** Communicative assertiveness index. Evaluates direct versus mitigated style:
  - **Adds:** num_imperatives (+1), has_disagreement (+1), baseline (+2)
  - **Subtracts:** has_hedge (-1), has_apology (-1)
- **Interpretation:** High values indicate direct and assertive style; low values indicate tentative and mitigated style.

### `interruption_climate`
- **Type:** Float
- **Range:** 0.0 to 1.0
- **Definition:** Density of interruptions in the immediate context, calculated as the proportion of interventions with `interrupts_previous = True` in a rolling window of the previous 5 turns.
- **Interpretation:** Captures whether the session moment is characterized by high or low frequency of interruptions.

### `climate_type`
- **Type:** Categorical
- **Values:** `calm`, `neutral`, `hostile`
- **Definition:** Classification of conversational climate based on `interruption_climate`:
  - **calm:** interruption_climate < 0.1 (few interruptions)
  - **neutral:** 0.1 ≤ interruption_climate ≤ 0.3
  - **hostile:** interruption_climate > 0.3 (many interruptions)

---

## 10. Affective Analysis

### `sentiment`
- **Type:** Categorical
- **Values:** `POS` (positive), `NEG` (negative), `NEU` (neutral)
- **Definition:** Classification of the general affective tone of the intervention, determined via BERT sentiment analysis model (pysentimiento).

### `emotion`
- **Type:** Categorical
- **Values:** `joy`, `anger`, `sadness`, `fear`, `surprise`, `disgust`, `others`
- **Definition:** Predominant emotion detected in the intervention, classified via BERT emotional analysis model. The "others" category includes emotions not clearly classifiable into the six basic ones.

---

## 11. Derived Analysis Variables

### `parity_index`
- **Type:** Float
- **Range:** 0.0 to 1.0
- **Level:** Session
- **Definition:** Gender equity index in the distribution of speaking time within a session. Calculated as the ratio between the smaller and larger percentage of speaking time by gender. A value of 1.0 indicates perfect parity (50%-50%); values near 0 indicate extreme dominance of one gender.

### `influence_ratio`
- **Type:** Float
- **Range:** >0
- **Level:** Speaker/Gender
- **Definition:** Ratio between words introduced for the first time in the session and words that repeat vocabulary from others. Values >1 indicate tendency to introduce new vocabulary (thematic leadership); values <1 indicate tendency to follow others' vocabulary.

---

## 12. Cluster Analysis Variables (User Level)

### `user_id`
- **Type:** String
- **Format:** `{session}__{speaker}`
- **Level:** User
- **Definition:** Unique identifier for each participant within a session. Combines session and speaker label. The same speaker in different sessions is treated as distinct users.

### `is_top3_speaker`
- **Type:** Binary
- **Values:** 0, 1
- **Level:** User
- **Definition:** Indicates whether the user is among the first 3 distinct speakers in the session, ordered by `start_time`. A value of 1 defines participants who initiate the conversation, typically moderators or panelists.

### `min_turn_number`
- **Type:** Integer
- **Range:** ≥1
- **Level:** User
- **Definition:** Earliest turn in which the user first intervenes in the session. Low values indicate early participation (leadership roles); high values indicate late incorporation (audience).

### `cluster_kmeans`
- **Type:** Integer
- **Values:** 0, 1
- **Level:** User
- **Definition:** Cluster assignment obtained via K-Means (k=2) on 45 standardized features. The empirical interpretation of clusters is:
  - **Cluster 0 — Moderators/Panelists:** Users who speak first (median turn=3), produce more interventions, duration, and words. 58.9% are top-3 speakers.
  - **Cluster 1 — Public/Audience:** Users who intervene late (median turn=35), with lower volume but more apologies and sadness expressions. Only 0.7% are top-3 speakers.

### `pct_female_mods` / `pct_female_auds`
- **Type:** Float
- **Range:** 0.0 to 100.0
- **Level:** Session
- **Definition:** Percentage of women among moderators (Cluster 0) or audience (Cluster 1) of each session. Used to quantify the gender attraction effect.

---

# PART II: GLOSSARY OF TECHNICAL TERMS

---

## Audio Processing

### Speaker Diarization
Automatic process of segmenting an audio recording according to "who speaks when." The system identifies speaker changes and groups segments corresponding to each person, assigning anonymous labels (SPEAKER_01, SPEAKER_02, etc.).

### Speaker Embedding
Vector representation of the distinctive acoustic characteristics of a speaker, extracted via neural networks. These fixed-dimension vectors (typically 192-512 dimensions) capture the "vocal fingerprint" that allows distinguishing different people.

### ECAPA-TDNN
Neural network architecture specialized in voice embedding extraction. Combines 1D convolutional layers with attention mechanisms to capture spectral features at multiple temporal scales.

### EBU R128
Standard of the European Broadcasting Union for loudness normalization. Defines methods to measure and adjust the perceived audio level, using the LUFS unit (Loudness Units relative to Full Scale).

### LUFS (Loudness Units Full Scale)
Loudness measurement unit that considers human perception of volume, not just the physical amplitude of the signal. The EBU R128 standard specifies -23 LUFS as the target level for broadcast content.

### VAD (Voice Activity Detection)
Technique to automatically determine which segments of an audio signal contain human speech versus silence, background noise, or other non-vocal sounds.

### Fundamental Frequency (F0) / Pitch
Vibration frequency of the vocal cords during phonation, measured in Hertz (Hz). It is the acoustic correlate of low/high pitch perception. Shows sexual dimorphism: typically 85-180 Hz in adult males and 165-255 Hz in adult females.

### Whisper
Automatic speech recognition (ASR) model developed by OpenAI. Trained on massive multilingual data, it stands out for its robustness to variations in accent, noise, and audio quality.

### Pyannote.audio
Open-source library for speaker-centered audio analysis, including pre-trained models for diarization, voice activity detection, and speaker verification.

### wav2vec2
Transformer-type neural network architecture developed by Meta/Facebook for learning audio representations. Pre-trained in a self-supervised manner, it can be fine-tuned for specific tasks such as speech recognition or gender classification.

---

## Natural Language Processing (NLP)

### Tokenization
Process of dividing a text into minimal units (tokens), typically words and punctuation marks.

### Lemmatization
Process of reducing words to their base form or lemma (e.g., "running", "ran", "runs" → "run"). Allows grouping morphological variants of the same word.

### POS Tagging (Part-of-Speech Tagging)
Automatic grammatical tagging that assigns to each word its morphosyntactic category (noun, verb, adjective, etc.).

### Dependency Parsing
Syntactic analysis that identifies grammatical relationships between words (subject, object, modifier, etc.) in the form of a dependency tree.

### spaCy
Open-source NLP library optimized for production, offering tokenization, lemmatization, POS tagging, entity recognition, and dependency parsing.

### Type-Token Ratio (TTR)
Measure of lexical diversity calculated as the proportion of unique words (types) over total words (tokens). Sensitive to text length: longer texts tend to have lower TTR.

### Jaccard Coefficient
Similarity measure between two sets, calculated as the size of the intersection divided by the size of the union. Range: 0 (disjoint sets) to 1 (identical sets).

### Hedge
In pragmatics, a linguistic expression that reduces the force or certainty of an utterance. Functions as negative politeness (protecting the interlocutor's face) or expression of epistemic uncertainty.

### pysentimiento
Sentiment and emotion analysis library based on Transformer models (BERT) fine-tuned for Spanish and English. Provides polarity classification (positive/negative/neutral) and discrete emotions.

### BERT (Bidirectional Encoder Representations from Transformers)
Language model architecture that processes text bidirectionally, capturing both previous and subsequent context for each word. Foundation of numerous state-of-the-art NLP models.

---

## Inferential Statistics

### Mann-Whitney U Test
Non-parametric test for comparing two independent groups. Evaluates whether one sample tends to have larger values than the other, without assuming normal distribution. Non-parametric equivalent of the t-test.

### Shapiro-Wilk Test
Normality test that evaluates whether a sample comes from a normal distribution. A low p-value (typically <0.05) indicates significant deviation from normality.

### Levene's Test
Homoscedasticity test that evaluates whether two or more groups have equal variances. More robust than Bartlett's test to deviations from normality.

### Chi-square Test (χ²)
Test to evaluate association between two categorical variables. Compares observed frequencies with those expected under independence.

### FDR (False Discovery Rate)
Expected proportion of false positives among all results declared significant. Alternative to familywise error rate (FWER) control offering greater power.

### Benjamini-Hochberg Correction
Procedure to control FDR in multiple comparisons. Orders p-values and applies adaptive thresholds that maintain the false discovery rate under control.

### Hedges' g
Effect size measure for differences between means, similar to Cohen's d but with correction for bias in small samples. Conventional interpretation: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.

### Cramér's V
Association measure for contingency tables, normalized between 0 and 1 regardless of table size. Based on the chi-square statistic.

### Cohen's d
Effect size measure expressing the difference between two means in units of combined standard deviation. Values: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 moderate, >0.8 large. Used in the cluster × gender cross to quantify the magnitude of differences between men and women within each cluster.

### Silhouette Score
Clustering quality measure evaluating how well each point is assigned to its cluster. Range: -1 to +1. Values close to 1 indicate dense and well-separated clusters; values close to 0 indicate overlapping; negative values indicate incorrect assignment.

### Confidence Interval (CI)
Range of values that, with a certain confidence level (typically 95%), is expected to contain the true population parameter. A CI crossing zero for a difference indicates no statistical significance.

### Bootstrap
Resampling method that estimates the sampling distribution of a statistic by generating multiple samples with replacement from the original dataset. Useful for constructing confidence intervals without parametric assumptions.

### Wilson Score Interval
Method for calculating confidence intervals for proportions that offers better coverage than the Wald interval, especially with proportions near 0 or 1 or small samples.

---

## Mixed Effects Models

### Mixed-Effects Model
Statistical model that includes both fixed effects (factors of interest whose effect is estimated) and random effects (factors representing a sample from a broader population). Appropriate for data with hierarchical structure or repeated measures.

### Fixed Effect
In mixed models, a parameter representing the average effect of a predictor variable in the population. It is the coefficient of interest that is reported and interpreted.

### Random Effect
In mixed models, a parameter capturing variability between grouping units (e.g., sessions, subjects). Allows the intercept or slope to vary between groups.

### ICC (Intraclass Correlation Coefficient)
Proportion of total variance explained by differences between groups (e.g., sessions). ICC = σ²_between / (σ²_between + σ²_within). Values close to 0 indicate that most variability is intra-group; values close to 1 indicate high intra-group homogeneity.

### LMM (Linear Mixed Model)
Mixed effects model for continuous dependent variables. Assumes normal errors and linear relationship between predictors and outcome.

### GLMM (Generalized Linear Mixed Model)
Extension of LMM for non-normal dependent variables (binary, counts, etc.) through appropriate link functions (logit for binary, log for counts).

### Link Function
In generalized linear models, the function that connects the linear predictor with the mean of the response distribution. For binary data, the logit function transforms probabilities into log-odds.

---

## Predictive Modeling

### Logistic Regression
Model to predict the probability of a binary outcome. Estimates log-odds as a linear function of predictors. Coefficients are interpreted in terms of odds ratios.

### Odds Ratio (OR)
Ratio of probabilities. OR = 1 indicates no effect; OR > 1 indicates the predictor increases the probability of the outcome; OR < 1 indicates it decreases it. Interpreted as the multiplicative change in odds per unit change in the predictor.

### AUC (Area Under the Curve)
Area under the ROC (Receiver Operating Characteristic) curve. Measures the discriminative capacity of a classification model. Range: 0.5 (chance) to 1.0 (perfect discrimination). Values >0.7 are considered acceptable; >0.8 good.

### ROC Curve
Graph plotting the true positive rate (sensitivity) versus the false positive rate (1-specificity) for different classification thresholds.

### Cross-Validation
Technique to evaluate a model's generalization capacity by systematically dividing data into training and test subsets.

---

## Cluster Analysis (Unsupervised Learning)

### K-Means
Partitional clustering algorithm that divides N observations into K groups, minimizing intra-cluster variance (inertia). Requires specifying K in advance. In this study, K=2 was selected as optimal by the Silhouette criterion.

### DBSCAN (Density-Based Spatial Clustering)
Density-based clustering algorithm that identifies groups as dense regions separated by low-density regions. Does not require specifying the number of clusters and can detect outliers (points not assigned to any cluster).

### Hierarchical Clustering
Agglomerative method that builds a hierarchy of clusters by iteratively merging the closest pairs. Produces a dendrogram visualizing the clustering structure at multiple levels of granularity.

### PCA (Principal Component Analysis)
Dimensionality reduction technique that projects high-dimensional data into a lower-dimensional space, preserving the maximum possible variance. Each principal component is a linear combination of the original variables.

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
Non-linear dimensionality reduction technique optimized for visualizing high-dimensional data in 2D or 3D. Preserves local structure (neighborhoods) at the cost of distorting global distances.

### ARI (Adjusted Rand Index)
Measure of concordance between two data partitions, adjusted for chance. Range: -1 to 1. Values close to 1 indicate the two groupings are very similar; 0 indicates chance-level concordance.

### Gender Attraction Effect
Phenomenon observed in this study whereby the gender composition of moderators/panelists (Cluster 0) of a session correlates positively with the gender composition of the audience (Cluster 1). Sessions with predominantly female moderators attract 51.6% female audience, compared to 27.2% in sessions with predominantly male moderators (Spearman ρ=0.361, p=0.004).

---

## Conversational Analysis

### Speaking Turn
Period during which a participant has the recognized right to speak. The turn-taking system organizes participant alternation in the conversation.

### Overlap
Simultaneous speech of two or more participants. Can be competitive (struggle for the turn) or collaborative (completing utterances, showing agreement).

### Interruption
Turn initiation by a speaker while another still has the floor, typically with the intention of taking the turn. Distinguished from collaborative overlap by its intrusive character.

### Backchannel
Brief verbal signals from the listener indicating attention and comprehension without intending to take the turn. Examples: "mhm", "yeah", "right", "I see".

### Response Latency
Time between the end of one speaker's turn and the start of the next speaker's turn. Very short or negative latencies may indicate anticipation or interruption.

### Idea Appropriation
Phenomenon where a participant takes up or reformulates ideas expressed by another without attributing their origin. In the study, operationalized as high lexical echoing without explicit attribution.

### Mansplaining
Colloquial term describing a pattern where a man explains something to a woman in a condescending manner, assuming lack of knowledge. In the study, operationalized symmetrically as "explaining pattern" applicable to both gender directions.

---

## Domain-Specific Terms

### Scientific Debate
Academic session format where multiple experts discuss a topic, typically with contrasting or complementary positions, moderated by a facilitator.

### Critical Care Medicine
Medical specialty dedicated to the diagnosis and treatment of life-threatening conditions requiring vital support and intensive monitoring (ICU).

### Gender Parity
Equitable representation of men and women, typically expressed as a 50%-50% proportion in panel composition, speaking time, or other participation metrics.

### Sticky Floor
Workplace metaphor describing invisible barriers that hinder initial advancement from entry-level positions. In the study, adapted to refer to the time each gender takes to first speak in a session.

### Attraction Effect
Effect whereby the representation of one gender in leadership or visibility positions (moderators, panelists) influences the gender composition of subsequent participation (audience). Empirically documented in this study with OR=3.16.

### Backlash
Negative reaction or social penalty that individuals (especially women) may face for behaviors that violate traditional gender expectations, such as assertiveness or directness.

---

*Document prepared to ensure replicability and complete understanding of the study.*