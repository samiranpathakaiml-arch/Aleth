"""
Generates aleth/data/task_hard.json
Run: python generate_hard.py
"""
import json, pathlib

# ── PAPERS ──────────────────────────────────────────────────────────────────
# 10 full_text | 10 abstract_only | 5 unavailable

FT = "full_text"; AO = "abstract_only"; UN = "unavailable"

def p(id_, title, abstract, full_text, year, authors, access):
    return {id_: {"id": id_, "title": title, "abstract": abstract,
                  "full_text": full_text, "year": year,
                  "authors": authors, "access_level": access}}

papers = {}

# ── full_text (10) ───────────────────────────────────────────────────────────
papers.update(p(
    "lecun1989",
    "Handwritten Digit Recognition with a Back-Propagation Network",
    "We apply backpropagation to train a convolutional network that recognises handwritten zip codes. The network learns hierarchical feature detectors automatically, achieving high accuracy on digit classification.",
    "ABSTRACT: A convolutional network trained with backpropagation achieves 99% accuracy on handwritten digit recognition.\n\nARCHITECTURE: The network uses shared weights in convolutional layers to learn local feature detectors, followed by subsampling layers, and fully connected classification layers.\n\nRESULTS: Tested on USPS zip-code dataset only. The results demonstrate viability for this specific character recognition task on this specific dataset.\n\nCONCLUSION: Convolutional networks can learn hierarchical features for handwritten character recognition on the task and dataset studied.",
    1989, ["Yann LeCun", "Bernhard Boser", "John S. Denker"], FT))

papers.update(p(
    "goodfellow2016dl",
    "Deep Learning (Textbook)",
    "A comprehensive textbook on deep learning methods covering feedforward networks, regularization, optimization, sequence modeling, and generative models.",
    "OVERVIEW: This book provides a broad survey of deep learning methods. No single empirical result is claimed as universally state-of-the-art.\n\nKEY THEME: Deep learning excels in specific domains (vision, speech, NLP) but the book cautions against overgeneralisation. Each method must be evaluated on its target task.\n\nNOTABLE CAVEAT: The authors explicitly note that no single architecture dominates all tasks and that empirical evaluation on the target domain is essential.\n\nCONCLUSION: Deep learning is a powerful family of methods, not a universal solution.",
    2016, ["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"], FT))

papers.update(p(
    "silver2017alphago",
    "Mastering the Game of Go without Human Knowledge",
    "AlphaGo Zero learns to play Go at superhuman level through self-play reinforcement learning, starting from random play and using no human game data.",
    "ABSTRACT: AlphaGo Zero achieves superhuman Go performance using self-play and a single neural network, with no human knowledge beyond game rules.\n\nMETHOD: Self-play reinforcement learning with Monte Carlo Tree Search. The policy and value networks are combined into a single residual network.\n\nSCOPE: Results are specific to the game of Go. The paper does not claim this method is state-of-the-art for all sequential decision problems — it is evaluated only on Go.\n\nRESULTS: Defeats AlphaGo Lee 100-0 and AlphaGo Master 89-11.\n\nCONCLUSION: Self-play RL is highly effective for two-player perfect-information board games, specifically Go.",
    2017, ["David Silver", "Julian Schrittwieser", "Karen Simonyan"], FT))

papers.update(p(
    "chen2020simclr",
    "A Simple Framework for Contrastive Learning of Visual Representations",
    "SimCLR achieves strong results on ImageNet linear evaluation using contrastive self-supervised learning with large batch sizes and strong data augmentation.",
    "ABSTRACT: SimCLR achieves 76.5% top-1 accuracy on ImageNet linear evaluation, competitive with supervised methods on this specific benchmark.\n\nMETHOD: Contrastive loss with data augmentation, a projection head, and large batch sizes.\n\nSCOPE: Evaluated primarily on ImageNet linear evaluation protocol. The paper does not establish SimCLR as the definitive solution to representation learning across all domains or tasks.\n\nRESULTS: 76.5% top-1 on ImageNet (linear eval), outperforming prior self-supervised methods on this benchmark only.\n\nCONCLUSION: SimCLR is a strong self-supervised approach for visual representation learning on ImageNet-scale data.",
    2020, ["Ting Chen", "Simon Kornblith", "Mohammad Norouzi", "Geoffrey Hinton"], FT))

papers.update(p(
    "devlin2019bert_med",
    "BioBERT: a pre-trained biomedical language representation model",
    "BioBERT applies BERT pre-training to biomedical text and improves NLP task scores on benchmark datasets. NLP metric improvements are demonstrated; direct causal links to patient outcomes are not established.",
    "ABSTRACT: BioBERT improves biomedical NLP metrics on named entity recognition, relation extraction, and QA benchmarks.\n\nMETHOD: BERT pre-trained on PubMed abstracts and PMC full-text articles.\n\nFINDINGS: BioBERT improves model performance metrics (F1, accuracy) on benchmark datasets. The paper studies NLP task performance, not clinical outcomes.\n\nLIMITATION: The paper does not measure patient outcomes. Improved NLP metrics do not imply improved clinical decisions or patient outcomes.\n\nCONCLUSION: Pre-training on biomedical text improves NLP benchmark scores; causation of improved patient outcomes is not studied.",
    2019, ["Jinhyuk Lee", "Wonjin Yoon", "Sungdong Kim", "Donghyeon Kim"], FT))

papers.update(p(
    "marcus2018critique",
    "Deep Learning: A Critical Appraisal",
    "Marcus critically examines limitations of deep learning, arguing it fails at compositionality, systematic generalization, and robustness, and that current results do not support claims of general intelligence.",
    "ABSTRACT: This paper argues that deep learning faces fundamental limitations including brittleness, inability to reason systematically, and poor generalisation outside training distribution.\n\nKEY ARGUMENTS: (1) DL lacks compositionality. (2) DL is data-hungry. (3) DL struggles out-of-distribution. (4) Current DL does not imply AGI.\n\nCONCLUSION: Marcus concludes deep learning has NOT definitively solved compositionality or systematic generalisation; significant challenges remain.",
    2018, ["Gary Marcus"], FT))

papers.update(p(
    "kohavi1995cv",
    "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection",
    "Empirical study comparing cross-validation strategies on UCI datasets. Stratified 10-fold cross-validation is recommended for small datasets; results do not generalise to all benchmarks.",
    "ABSTRACT: We study cross-validation and bootstrap methods empirically on a set of UCI datasets.\n\nFINDINGS: Stratified 10-fold CV provides a good balance of bias and variance on the small datasets tested. Results show variance is high for 2-fold but low for 10-fold.\n\nSCOPE: Results are based on UCI repository datasets of the era. The paper does not claim these findings generalise to all real-world machine learning scenarios.\n\nCONCLUSION: Stratified 10-fold CV is recommended for the datasets studied; generalisation to all ML benchmarks requires separate empirical validation.",
    1995, ["Ron Kohavi"], FT))

papers.update(p(
    "sculley2015debt",
    "Hidden Technical Debt in Machine Learning Systems",
    "Identifies numerous sources of hidden technical debt specific to ML systems: entanglement, correction cascades, undeclared consumers, data dependencies, and configuration debt.",
    "ABSTRACT: ML systems accumulate technical debt faster than traditional software due to unique factors.\n\nKEY FINDINGS: (1) CACE principle: Changing Anything Changes Everything — entanglement makes ML systems hard to maintain. (2) Correction cascades create brittle systems. (3) Undeclared consumers create tight coupling. (4) Data dependencies are more dangerous than code dependencies.\n\nCONCLUSION: Machine learning systems are NOT easy to maintain; they carry significant hidden technical debt that is costly to identify and pay down.",
    2015, ["D. Sculley", "Gary Holt", "Daniel Golovin", "Eugene Davydov", "Todd Phillips"], FT))

papers.update(p(
    "zech2018pneumonia",
    "Variable generalization performance of a deep learning model to detect pneumonia from chest radiographs",
    "A deep learning model trained on CheXNet data achieves high AUC within a hospital system but performance degrades significantly when tested across different hospital systems, revealing site-specific confounders.",
    "ABSTRACT: A pneumonia detection model trained at one hospital shows high AUC within that system but significantly lower AUC when tested at other hospitals.\n\nKEY FINDING: Hospital-specific image acquisition characteristics act as confounders. The model learns spurious correlations specific to scanner/site characteristics rather than purely pathological features.\n\nCONCLUSION: Chest X-ray AI models do NOT generalise reliably across hospital sites. This paper explicitly demonstrates the failure of cross-site generalisation, contradicting claims of universal applicability.",
    2018, ["John R. Zech", "Marcus A. Badgeley", "Manway Liu", "Anthony B. Costa"], FT))

papers.update(p(
    "shah2021bias",
    "Systematic evaluation of AI in healthcare: biases, disparities, and recommendations",
    "Identifies associations between algorithmic deployment and health disparities, finding that AI systems used in clinical settings correlate with worse outcomes for minority groups. Causal mechanisms are not established.",
    "ABSTRACT: We identify systematic associations between healthcare AI deployment and health outcome disparities across racial and socioeconomic groups.\n\nKEY FINDING: AI systems show statistically significant correlations with disparate outcomes. However, the study design is observational; causal pathways are not established.\n\nEXPLICIT LIMITATION: Authors caution that observed associations do not establish causation. Multiple confounders exist. Rigorous RCTs are needed to determine causal impact.\n\nCONCLUSION: Association between AI bias and health disparities is documented; direct causation has NOT been established by this study.",
    2021, ["Nigam H. Shah", "Emily Enright", "Keith Humphreys"], FT))

# ── abstract_only (10) ───────────────────────────────────────────────────────
papers.update(p(
    "obermeyer2019bias",
    "Dissecting racial bias in an algorithm used to manage the health of populations",
    "A widely-used commercial healthcare prediction algorithm systematically underestimates illness severity for Black patients compared to White patients at the same predicted risk score, resulting in Black patients being less likely to be referred to high-risk care programs.",
    None, 2019, ["Ziad Obermeyer", "Brian Powers", "Christine Vogeli", "Sendhil Mullainathan"], AO))

papers.update(p(
    "rajpurkar2017chexnet",
    "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays Using Deep Learning",
    "CheXNet, a 121-layer DenseNet trained on ChestX-ray14, exceeds the performance of radiologists on pneumonia detection as measured by F1 score on a held-out test set from the same distribution.",
    None, 2017, ["Pranav Rajpurkar", "Jeremy Irvin", "Kaylie Zhu", "Brandon Yang"], AO))

papers.update(p(
    "esteva2017derm",
    "Dermatologist-level classification of skin cancer with deep neural networks",
    "A CNN trained on 129,450 clinical images classifies skin lesions at a level of competence comparable to dermatologists on two specific classification tasks using test images from the same distribution as training.",
    None, 2017, ["Andre Esteva", "Brett Kuprel", "Roberto A. Novoa", "Justin Ko"], AO))

papers.update(p(
    "topol2019medicine",
    "High-performance medicine: the convergence of human and artificial intelligence",
    "A review arguing that AI will transform medicine by improving diagnostic accuracy and personalising treatment, but emphasising the need for prospective clinical trials, regulatory oversight, and human-AI collaboration before deployment.",
    None, 2019, ["Eric J. Topol"], AO))

papers.update(p(
    "weng2017cardio",
    "Can machine-learning improve cardiovascular risk prediction using routine clinical data?",
    "Machine learning models including neural networks improve C-statistic for cardiovascular risk prediction compared to established clinical scores on the studied cohort. The study measures a surrogate metric (C-statistic) and does not measure actual cardiovascular events prevented.",
    None, 2017, ["Stephen F. Weng", "Jenna Reps", "Joe Kai", "Jonathan M. Garibaldi"], AO))

papers.update(p(
    "johnson2016mimic",
    "MIMIC-III, a freely accessible critical care database",
    "MIMIC-III is a large, freely available database comprising de-identified health-related data associated with over 40,000 patients who stayed in critical care units at Beth Israel Deaconess Medical Center between 2001 and 2012.",
    None, 2016, ["Alistair E.W. Johnson", "Tom J. Pollard", "Lu Shen", "Li-wei H. Lehman"], AO))

papers.update(p(
    "beam2018clinical",
    "Clinical concept embeddings learned from massive sources of multimodal medical data",
    "Word embeddings trained on clinical notes from a large EHR exhibit semantic structure. The paper studies embedding quality on NLP benchmarks, not clinical outcomes.",
    None, 2018, ["Andrew L. Beam", "Benjamin Kompa", "Allen Schmaltz", "Samuel Finlayson"], AO))

papers.update(p(
    "tschandl2019ham",
    "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of pigmented skin lesions",
    "HAM10000 provides 10,015 dermatoscopic images from two sites for machine learning research in skin lesion classification. It is a benchmark dataset, not a clinical deployment validation.",
    None, 2019, ["Philipp Tschandl", "Cliff Rosendahl", "Harald Kittler"], AO))

papers.update(p(
    "litjens2017survey",
    "A survey on deep learning in medical image analysis",
    "Survey of deep learning applications across medical imaging modalities. Deep learning achieves strong results in specific tasks on specific datasets; the survey notes highly variable performance and the absence of universal superiority over traditional methods.",
    None, 2017, ["Geert Litjens", "Thijs Kooi", "Babak Ehteshami Bejnordi", "Arnaud Arindra Aditi Setio"], AO))

papers.update(p(
    "shen2017medical",
    "Deep Learning in Medical Image Analysis",
    "Reviews deep learning for radiology and pathology tasks. Concludes that performance is task- and dataset-specific and that prospective validation in clinical workflows is largely absent.",
    None, 2017, ["Dinggang Shen", "Guorong Wu", "Heung-Il Suk"], AO))

# ── unavailable (5) ──────────────────────────────────────────────────────────
papers.update(p(
    "fda2021guidance",
    "Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) Action Plan",
    "FDA regulatory framework document on review and oversight requirements for AI/ML-based medical devices. Not publicly accessible as a citable research paper.",
    None, 2021, ["U.S. Food and Drug Administration"], UN))

papers.update(p(
    "who2021ethics",
    "Ethics and governance of artificial intelligence for health",
    "WHO guidance document outlining six ethical principles for AI in health and calling for strong governance, regulatory oversight, and human accountability in AI deployment. Access restricted.",
    None, 2021, ["World Health Organization"], UN))

papers.update(p(
    "internal2019rct",
    "Randomized Controlled Trial: AI-Assisted Diagnosis in ICU Settings (Confidential)",
    "Confidential proprietary RCT data. Results were inconclusive (p=0.43); the AI-assisted arm did not show statistically significant improvement over standard care. Data not publicly available.",
    None, 2019, ["[Confidential Authors]"], UN))

papers.update(p(
    "preprint2022llm",
    "Large Language Models for Clinical Diagnosis: A Preliminary Report (Retracted)",
    "Retracted preprint claiming LLMs match specialist physicians in diagnosis. Retraction notice: analysis contained errors in patient cohort selection and statistical testing. Results cannot be relied upon.",
    None, 2022, ["[Authors redacted post-retraction]"], UN))

papers.update(p(
    "proprietary2021clinical",
    "Proprietary Clinical Trial: AI Diagnostic System Validation (Industry Confidential)",
    "Industry-confidential clinical trial data. Not peer-reviewed or publicly available. Results disputed by independent reviewers.",
    None, 2021, ["[Proprietary — Not Disclosed]"], UN))

# ── CLAIMS ───────────────────────────────────────────────────────────────────
# Drift A = correlation→causation (0.3) | B = surrogate→hard outcome (0.2)
# Drift C = overgeneralisation (0.4)   | D = mirror lie (0.0)
# Clean = no drift (0.9-1.0)

claims = [
    # ── DRIFT A: Correlation → Causation (6 claims) ─────────────────────────
    {"id":"claim_h01","text":"Algorithmic bias in healthcare AI directly causes disparate health outcomes for minority patients.","citations":["obermeyer2019bias","shah2021bias","fda2021guidance"],"context":"Healthcare AI systems have been scrutinised for fairness. Studies show algorithmic bias directly causes worse health outcomes for minority groups.","claim_type":"empirical"},
    {"id":"claim_h02","text":"Using BioBERT on clinical notes directly causes improved patient outcomes in hospitals.","citations":["devlin2019bert_med","beam2018clinical","johnson2016mimic"],"context":"NLP tools applied to EHRs are expected to improve care. BioBERT applied to clinical NLP directly causes improved patient outcomes.","claim_type":"empirical"},
    {"id":"claim_h03","text":"Higher benchmark accuracy on ImageNet directly causes better diagnostic performance in real hospital deployments.","citations":["lecun1989","rajpurkar2017chexnet","zech2018pneumonia"],"context":"ImageNet accuracy is used as a proxy for medical AI quality. Higher ImageNet accuracy directly causes better diagnostic outcomes in hospitals.","claim_type":"empirical"},
    {"id":"claim_h04","text":"Screen time in patients correlating with poor health metrics proves that digital device usage directly causes cognitive decline.","citations":["shah2021bias","obermeyer2019bias","topol2019medicine"],"context":"Observational studies identify correlates of cognitive decline. Device screen time directly causes cognitive decline based on the cited associations.","claim_type":"empirical"},
    {"id":"claim_h05","text":"Deploying deep learning in radiology departments directly causes a reduction in radiologist workload and burnout.","citations":["rajpurkar2017chexnet","litjens2017survey","zech2018pneumonia"],"context":"AI adoption in radiology is expected to change workflows. Deep learning deployment in radiology directly causes reduced radiologist workload.","claim_type":"empirical"},
    {"id":"claim_h06","text":"Cardiovascular risk score improvement observed in ML studies proves that deploying these models directly causes fewer cardiac events in populations.","citations":["weng2017cardio","topol2019medicine","who2021ethics"],"context":"ML cardiovascular risk tools have shown improved metrics. Improved risk scores directly cause fewer cardiac events when deployed.","claim_type":"empirical"},

    # ── DRIFT B: Surrogate → Hard Outcome (6 claims) ────────────────────────
    {"id":"claim_h07","text":"SimCLR's improved ImageNet linear evaluation accuracy proves it will outperform supervised methods in all real-world deployment scenarios.","citations":["chen2020simclr","goodfellow2016dl","kohavi1995cv"],"context":"Self-supervised learning benchmarks on ImageNet are used to predict production performance. SimCLR's benchmark gains prove superior real-world performance.","claim_type":"empirical"},
    {"id":"claim_h08","text":"AI dermatology models achieving AUC greater than 0.95 on held-out test sets will prevent skin cancer deaths in clinical practice.","citations":["esteva2017derm","tschandl2019ham","fda2021guidance"],"context":"High AUC in skin lesion classification is cited as evidence of life-saving potential. AUC above 0.95 on test sets proves that models prevent cancer deaths.","claim_type":"empirical"},
    {"id":"claim_h09","text":"Deep learning models that reduce false-positive rates in chest X-ray studies will prevent unnecessary patient hospitalisations at scale.","citations":["rajpurkar2017chexnet","zech2018pneumonia","internal2019rct"],"context":"Reducing false positives is considered a pathway to better care. Reduced FP rate on test datasets prevents unnecessary hospitalisations.","claim_type":"empirical"},
    {"id":"claim_h10","text":"Machine learning models improving the C-statistic for cardiovascular risk prediction will prevent heart attacks when deployed in clinical care.","citations":["weng2017cardio","topol2019medicine","who2021ethics"],"context":"Improved C-statistic is equated with mortality benefit. A better C-statistic proves the model will prevent heart attacks in deployed populations.","claim_type":"empirical"},
    {"id":"claim_h11","text":"A model achieving 99% accuracy on the MIMIC-III benchmark will successfully and reliably diagnose ICU patients in any hospital setting.","citations":["johnson2016mimic","shah2021bias","preprint2022llm"],"context":"High MIMIC-III accuracy is treated as a guarantee of clinical success. 99% benchmark accuracy ensures reliable ICU diagnosis anywhere.","claim_type":"empirical"},
    {"id":"claim_h12","text":"Reducing mathematical bias metrics in AI algorithms ensures equitable health outcomes for all patient groups in practice.","citations":["obermeyer2019bias","shah2021bias","who2021ethics"],"context":"Algorithmic fairness metrics are presented as sufficient for health equity. Reducing bias metrics ensures equitable health outcomes.","claim_type":"empirical"},

    # ── DRIFT C: Overgeneralisation (6 claims) ───────────────────────────────
    {"id":"claim_h13","text":"AlphaGo Zero's self-play reinforcement learning approach is the state-of-the-art method for all sequential decision-making problems.","citations":["silver2017alphago","goodfellow2016dl","marcus2018critique"],"context":"AlphaGo Zero's Go success is widely cited. Its self-play approach is now universally the best method for sequential decision problems.","claim_type":"empirical"},
    {"id":"claim_h14","text":"Deep learning is unambiguously superior to all traditional machine learning methods across every medical imaging task and dataset.","citations":["litjens2017survey","shen2017medical","marcus2018critique"],"context":"Deep learning dominates computer vision. It is universally superior to all traditional methods for all medical imaging tasks.","claim_type":"empirical"},
    {"id":"claim_h15","text":"SimCLR's contrastive learning framework is the definitive universal solution to representation learning across all domains and modalities.","citations":["chen2020simclr","goodfellow2016dl","sculley2015debt"],"context":"SimCLR's ImageNet success is extrapolated broadly. It is the universal solution to representation learning for all domains.","claim_type":"theoretical"},
    {"id":"claim_h16","text":"Cross-validation techniques proven effective on UCI datasets from 1995 generalise to and are optimal for all modern real-world machine learning applications.","citations":["kohavi1995cv","sculley2015debt","marcus2018critique"],"context":"Classic cross-validation studies are applied to all settings. Kohavi (1995) findings apply universally to all modern ML applications.","claim_type":"methodological"},
    {"id":"claim_h17","text":"CheXNet's radiologist-level pneumonia detection capability generalises to all hospital environments and imaging systems worldwide.","citations":["rajpurkar2017chexnet","zech2018pneumonia","litjens2017survey"],"context":"CheXNet's benchmark result is treated as universally applicable. Its performance generalises to any hospital worldwide.","claim_type":"empirical"},
    {"id":"claim_h18","text":"BERT pre-trained on Wikipedia and Books creates universal medical knowledge that enables reliable clinical decision-making without fine-tuning.","citations":["devlin2019bert_med","beam2018clinical","topol2019medicine"],"context":"BERT's general pre-training is assumed sufficient for clinical tasks. General BERT pre-training creates universal medical knowledge for clinical decisions.","claim_type":"theoretical"},

    # ── DRIFT D: Mirror Lie (6 claims) ───────────────────────────────────────
    {"id":"claim_h19","text":"Zech et al. (2018) demonstrated that chest X-ray AI models are robust and generalise reliably across different hospital systems.","citations":["zech2018pneumonia","rajpurkar2017chexnet","fda2021guidance"],"context":"Cross-site robustness of medical AI is frequently cited. Zech et al. showed these models generalise reliably across hospital systems.","claim_type":"empirical"},
    {"id":"claim_h20","text":"Marcus (2018) concluded that deep learning has definitively solved the challenges of compositionality and systematic generalisation in AI.","citations":["marcus2018critique","goodfellow2016dl","sculley2015debt"],"context":"Marcus (2018) is cited as validating deep learning capabilities. The paper concluded deep learning has solved compositionality.","claim_type":"theoretical"},
    {"id":"claim_h21","text":"Sculley et al. (2015) found that machine learning systems are straightforward to maintain and accumulate less technical debt than traditional software.","citations":["sculley2015debt","kohavi1995cv","goodfellow2016dl"],"context":"ML system maintainability is often assumed. Sculley et al. found ML systems are easy to maintain with low technical debt.","claim_type":"empirical"},
    {"id":"claim_h22","text":"The WHO (2021) ethics guidelines concluded that AI systems should be deployed in healthcare settings without the need for regulatory oversight.","citations":["who2021ethics","topol2019medicine","fda2021guidance"],"context":"International AI governance documents shape deployment. WHO guidelines concluded AI needs no regulatory oversight in healthcare.","claim_type":"theoretical"},
    {"id":"claim_h23","text":"Obermeyer et al. (2019) found that commercial healthcare prediction algorithms show no significant racial bias in patient risk assignment.","citations":["obermeyer2019bias","shah2021bias","proprietary2021clinical"],"context":"Healthcare AI fairness audits produce mixed findings. Obermeyer et al. found no significant racial bias in commercial algorithms.","claim_type":"empirical"},
    {"id":"claim_h24","text":"Confidential RCT data (2019) showed the AI diagnostic system significantly outperformed clinicians with strong statistical confidence (p < 0.01).","citations":["internal2019rct","rajpurkar2017chexnet","zech2018pneumonia"],"context":"Internal clinical trials validate AI systems before deployment. The 2019 RCT showed the system significantly outperformed clinicians.","claim_type":"empirical"},

    # ── CLEAN (6 claims, score 0.9–1.0) ──────────────────────────────────────
    {"id":"claim_h25","text":"LeNet demonstrated that convolutional neural networks can learn hierarchical features for handwritten character recognition on the USPS zip-code dataset.","citations":["lecun1989"],"context":"Early CNNs established foundational capabilities. LeNet showed CNNs learn hierarchical features for character recognition on specific datasets.","claim_type":"empirical"},
    {"id":"claim_h26","text":"SimCLR achieves 76.5% top-1 accuracy on the ImageNet linear evaluation benchmark using contrastive self-supervised learning with large batch sizes and strong augmentation.","citations":["chen2020simclr"],"context":"Self-supervised learning benchmarks on ImageNet track progress. SimCLR achieves 76.5% top-1 on the ImageNet linear evaluation.","claim_type":"empirical"},
    {"id":"claim_h27","text":"Zech et al. (2018) found that a pneumonia detection model trained on one hospital system performed significantly worse when tested on data from a different hospital system.","citations":["zech2018pneumonia"],"context":"Cross-site generalisation is a key challenge for medical AI. Zech et al. showed performance degrades substantially across hospital sites.","claim_type":"empirical"},
    {"id":"claim_h28","text":"AlphaGo Zero achieves superhuman Go performance through self-play reinforcement learning starting from random play, using no human game data.","citations":["silver2017alphago"],"context":"AlphaGo Zero is a landmark in reinforcement learning. It learns entirely from self-play without human game data.","claim_type":"empirical"},
    {"id":"claim_h29","text":"Sculley et al. (2015) identify that machine learning systems accumulate hidden technical debt from entanglement, correction cascades, and undeclared consumers.","citations":["sculley2015debt"],"context":"ML engineering challenges are studied systematically. Sculley et al. describe unique sources of technical debt in ML systems.","claim_type":"empirical"},
    {"id":"claim_h30","text":"Kohavi (1995) empirically shows that stratified 10-fold cross-validation provides lower variance than 2-fold cross-validation on the UCI datasets studied.","citations":["kohavi1995cv"],"context":"Cross-validation methodology matters for reliable evaluation. Kohavi (1995) compared variance of different CV strategies on UCI datasets.","claim_type":"methodological"},
]

# ── GROUND TRUTH ─────────────────────────────────────────────────────────────
def gt(cid, score, reasoning, primary, concepts, drift=True, chain=None):
    return {"claim_id":cid,"true_support_score":score,"reasoning":reasoning,
            "primary_evidence_paper":primary,"key_concepts":concepts,
            "has_citation_drift":drift,"drift_chain":chain or []}

ground_truth = [
    # DRIFT A — Correlation → Causation (score 0.3)
    gt("claim_h01", 0.3,
       "Shah (2021) and Obermeyer (2019) document ASSOCIATIONS between algorithmic bias and health disparities; neither establishes causal pathways. The claim replaces 'associated with' with 'directly causes' — a classic correlation-to-causation drift. Both papers explicitly caution that causation is not established.",
       "obermeyer2019bias", ["correlation not causation","association only","causal pathway not established","no RCT"]),
    gt("claim_h02", 0.3,
       "BioBERT improves NLP METRIC scores on biomedical benchmarks. The paper does not study patient outcomes. Claiming direct causation of improved patient outcomes from NLP metric gains is correlation-to-causation drift.",
       "devlin2019bert_med", ["NLP metrics not patient outcomes","correlation not causation","benchmark performance","no clinical trial"]),
    gt("claim_h03", 0.3,
       "High ImageNet accuracy is CORRELATED with but does not causally determine hospital diagnostic performance. Zech (2018) explicitly shows models with high benchmark accuracy fail in cross-site deployment — the claim ignores this contradicting evidence.",
       "rajpurkar2017chexnet", ["correlation not causation","benchmark vs deployment","confounders","zech contradicts"]),
    gt("claim_h04", 0.3,
       "The cited papers study observational associations between variables; none establish causal mechanisms for cognitive decline from screen time. Correlation-to-causation drift.",
       "shah2021bias", ["correlation not causation","observational study","confounders","causation not proven"]),
    gt("claim_h05", 0.3,
       "Rajpurkar (2017) shows high accuracy on a test set; Zech (2018) shows cross-site failure. Neither paper studies radiologist workload. The causal claim about workload reduction is unsupported — correlation-to-causation drift.",
       "rajpurkar2017chexnet", ["correlation not causation","workload not studied","benchmark accuracy","deployment gap"]),
    gt("claim_h06", 0.3,
       "Weng (2017) measures improved C-statistic (a surrogate metric) in an observational study, NOT cardiac events prevented. The paper does not establish causal reduction of cardiac events. This is both correlation-to-causation and surrogate-to-hard-outcome drift.",
       "weng2017cardio", ["correlation not causation","C-statistic is surrogate","cardiac events not measured","observational study"]),

    # DRIFT B — Surrogate → Hard Outcome (score 0.2)
    gt("claim_h07", 0.2,
       "SimCLR's 76.5% ImageNet linear evaluation is a SURROGATE METRIC. The paper does not evaluate real-world deployment performance. Linear eval accuracy does not establish superiority across all deployment scenarios — surrogate-to-hard-outcome drift.",
       "chen2020simclr", ["surrogate metric","linear evaluation only","real-world not tested","overpromise from benchmark"]),
    gt("claim_h08", 0.2,
       "Esteva (2017) measures AUC on a held-out test set — a surrogate metric. Cancer mortality reduction (a hard outcome) is never measured. High test-set AUC does not prove mortality prevention — surrogate-to-hard-outcome drift.",
       "esteva2017derm", ["surrogate endpoint","AUC not mortality","held-out test set","survival not measured"]),
    gt("claim_h09", 0.2,
       "Papers measure false-positive rate on test datasets (surrogate). Hospitalisation rates (hard outcome) are never measured. Zech (2018) further shows cross-site failures that undermine the claim. Surrogate-to-hard-outcome drift.",
       "rajpurkar2017chexnet", ["surrogate endpoint","false positive rate","hospitalisation not measured","deployment failures"]),
    gt("claim_h10", 0.2,
       "Weng (2017) measures C-statistic improvement — a surrogate metric. Actual cardiac events prevented (hard outcome) are not measured in this study. Surrogate-to-hard-outcome drift from model metric to mortality claim.",
       "weng2017cardio", ["surrogate endpoint","C-statistic","cardiac events not measured","surrogate to hard outcome"]),
    gt("claim_h11", 0.2,
       "MIMIC-III benchmark accuracy is a surrogate metric for a single dataset from one hospital. Reliable ICU diagnosis across all hospitals (hard outcome) is not established. Preprint2022 is retracted and unreliable. Surrogate-to-hard-outcome drift.",
       "johnson2016mimic", ["surrogate metric","benchmark accuracy","hospital generalisation not proven","retracted preprint"]),
    gt("claim_h12", 0.2,
       "Mathematical bias metrics (surrogate) being reduced does not guarantee equitable health outcomes (hard outcome) in practice; multiple other factors determine outcomes. Surrogate-to-hard-outcome drift.",
       "obermeyer2019bias", ["surrogate metric","fairness metric not outcome","equitable outcomes not proven","surrogate to hard outcome"]),

    # DRIFT C — Overgeneralisation (score 0.4)
    gt("claim_h13", 0.4,
       "Silver (2017) demonstrates AlphaGo Zero's self-play for Go specifically. The paper explicitly scopes results to Go and does not claim state-of-the-art for all sequential decision problems. Marcus (2018) further cautions against such generalisations. Overgeneralisation drift.",
       "silver2017alphago", ["specific to Go","not universal","overgeneralisation","sequential decision problems not all studied"]),
    gt("claim_h14", 0.4,
       "Litjens (2017) and Shen (2017) surveys note highly variable performance and task-specific results. Marcus (2018) explicitly argues deep learning lacks universal superiority. 'Unambiguously superior for all tasks' is an overgeneralisation drift.",
       "litjens2017survey", ["variable performance","task specific","not universally superior","overgeneralisation","litjens contradicts"]),
    gt("claim_h15", 0.4,
       "Chen (2020) evaluates SimCLR on ImageNet linear evaluation and certain downstream tasks. Sculley (2015) warns that benchmark performance rarely transfers universally. 'Universal solution for all domains' is an overgeneralisation drift.",
       "chen2020simclr", ["ImageNet only","not all domains","overgeneralisation","benchmark does not generalise universally"]),
    gt("claim_h16", 0.4,
       "Kohavi (1995) explicitly scopes results to UCI datasets of that era. Sculley (2015) and Marcus (2018) highlight that evaluation methodology must be re-validated for each new context. Claiming universal optimality across all modern applications is an overgeneralisation drift.",
       "kohavi1995cv", ["UCI datasets only","1995 scope","not universal","overgeneralisation","modern applications differ"]),
    gt("claim_h17", 0.4,
       "Rajpurkar (2017) tests CheXNet on a specific held-out set from the same distribution. Zech (2018) directly contradicts universal applicability by showing significant performance degradation across hospital sites. 'All environments worldwide' is an overgeneralisation drift contradicted by Zech.",
       "rajpurkar2017chexnet", ["single distribution test","zech contradicts","hospital site variation","overgeneralisation","generalisation failure"]),
    gt("claim_h18", 0.4,
       "BioBERT and Beam (2018) improve NLP task metrics after domain-specific fine-tuning. General BERT without fine-tuning has weaker medical performance. Topol (2019) calls for prospective trials, not direct deployment. 'Universal medical knowledge without fine-tuning' is an overgeneralisation drift.",
       "devlin2019bert_med", ["fine-tuning required","not universal knowledge","clinical trials needed","overgeneralisation"]),

    # DRIFT D — Mirror Lie (score 0.0)
    gt("claim_h19", 0.0,
       "Zech (2018) found the OPPOSITE: models fail to generalise across hospital sites due to site-specific confounders. The claim directly inverts the paper's conclusion. This is a Mirror Lie drift — the cited source contradicts the claim entirely.",
       "zech2018pneumonia", ["mirror lie","paper says opposite","generalisation failure","confounders","inverted conclusion"]),
    gt("claim_h20", 0.0,
       "Marcus (2018) explicitly argues deep learning has NOT solved compositionality or systematic generalisation — the paper is a CRITIQUE. The claim inverts the paper's thesis. Mirror Lie drift.",
       "marcus2018critique", ["mirror lie","marcus is a critique","compositionality not solved","inverted conclusion","paper says opposite"]),
    gt("claim_h21", 0.0,
       "Sculley (2015) concludes ML systems are HARD to maintain and carry SIGNIFICANT hidden technical debt. The claim inverts this finding entirely. Mirror Lie drift.",
       "sculley2015debt", ["mirror lie","paper says high debt","maintenance is difficult","inverted conclusion","paper says opposite"]),
    gt("claim_h22", 0.0,
       "WHO (2021) calls for STRONG regulatory oversight and governance — the opposite of what the claim states. The paper is unavailable for direct verification, but the abstract confirms this direction. Mirror Lie drift on an unavailable source.",
       "who2021ethics", ["mirror lie","WHO requires oversight","regulatory governance","inverted conclusion","unavailable source"]),
    gt("claim_h23", 0.0,
       "Obermeyer (2019) found SIGNIFICANT racial bias systematically underestimating Black patient illness severity. The claim inverts this central finding. Mirror Lie drift.",
       "obermeyer2019bias", ["mirror lie","paper found significant bias","racial disparity","inverted conclusion","paper says opposite"]),
    gt("claim_h24", 0.0,
       "The confidential RCT (unavailable) reports INCONCLUSIVE results (p=0.43), not significant outperformance (p<0.01). The claim fabricates statistical significance from an unavailable inconclusive source. Mirror Lie drift on unavailable data.",
       "internal2019rct", ["mirror lie","inconclusive results","p=0.43 not significant","fabricated statistics","unavailable source"]),

    # CLEAN (score 0.9–1.0)
    gt("claim_h25", 1.0,
       "LeCun (1989) explicitly demonstrates CNN hierarchical feature learning for handwritten digit recognition on the USPS dataset. Claim is fully scoped and supported.",
       "lecun1989", ["convolutional","hierarchical features","USPS dataset","character recognition","backpropagation"], False),
    gt("claim_h26", 1.0,
       "Chen (2020) explicitly reports 76.5% top-1 accuracy on ImageNet linear evaluation using large batch contrastive learning and strong augmentation.",
       "chen2020simclr", ["76.5%","ImageNet linear evaluation","contrastive","large batch","data augmentation"], False),
    gt("claim_h27", 1.0,
       "Zech (2018) directly demonstrates cross-site performance degradation, with the key finding that models trained on one hospital fail at others.",
       "zech2018pneumonia", ["cross-site failure","hospital generalisation","performance degradation","site-specific confounders"], False),
    gt("claim_h28", 1.0,
       "Silver (2017) explicitly describes AlphaGo Zero learning from self-play with no human game data, achieving superhuman Go performance.",
       "silver2017alphago", ["self-play","no human data","superhuman","reinforcement learning","Go"], False),
    gt("claim_h29", 1.0,
       "Sculley (2015) explicitly names entanglement, correction cascades, and undeclared consumers as hidden technical debt sources in ML systems.",
       "sculley2015debt", ["entanglement","correction cascades","undeclared consumers","technical debt","hidden"], False),
    gt("claim_h30", 1.0,
       "Kohavi (1995) empirically shows 10-fold CV has lower variance than 2-fold on the UCI datasets, making it preferable for small datasets.",
       "kohavi1995cv", ["stratified 10-fold","lower variance","2-fold","UCI datasets","cross-validation"], False),
]

# ── ASSEMBLE ─────────────────────────────────────────────────────────────────
task = {"task_id":"hard","max_steps":150,"claims":claims,"papers":papers,"ground_truth":ground_truth}
out  = pathlib.Path("aleth/data/task_hard.json")
out.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Written {out}  ({out.stat().st_size:,} bytes)")
print(f"Claims : {len(claims)}")
print(f"Papers : {len(papers)}")
print(f"  full_text    : {sum(1 for p in papers.values() if p['access_level']=='full_text')}")
print(f"  abstract_only: {sum(1 for p in papers.values() if p['access_level']=='abstract_only')}")
print(f"  unavailable  : {sum(1 for p in papers.values() if p['access_level']=='unavailable')}")
multi = [c for c in claims if len(c["citations"]) >= 3]
drift = [g for g in ground_truth if g["has_citation_drift"]]
scores = {g["claim_id"]: g["true_support_score"] for g in ground_truth}
print(f"Multi-citation (3+): {len(multi)}")
print(f"Drifted claims     : {len(drift)}")
print(f"Score distribution : {sorted(set(scores.values()))}")
