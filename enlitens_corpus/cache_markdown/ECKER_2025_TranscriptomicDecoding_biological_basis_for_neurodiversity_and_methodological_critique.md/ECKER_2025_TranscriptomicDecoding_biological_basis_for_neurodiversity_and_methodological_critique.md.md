# Extracted Text for ECKER_2025_TranscriptomicDecoding_biological_basis_for_neurodiversity_and_methodological_critique.md.pdf

DOCUMENT SUMMARY

This highly technical neuroscience paper provides a powerful biological basis for Enlitens' core 
mission. It uses advanced brain imaging and gene expression analysis to demonstrate that 
"neurodiverse brain organization" is a biological reality. The research successfully moves 
beyond simple metrics, showing that individualized patterns of brain structure and gene 
expression can be used to subgroup individuals in behaviorally meaningful ways (e.g., anxiety 
levels), directly challenging a one-size-fits-all, categorical model of the human mind. The paper's
in-depth discussion of the limitations of various sophisticated statistical models also serves as a 
strong analogy for the inherent flaws and biases in standardized psychological testing.

FILENAME

ECKER_2025_TranscriptomicDecoding_biological_basis_for_neurodiversity_and_methodologic
al_critique.md

METADATA

● Primary Category: NEURODIVERSITY
● Document Type: research_article
● Relevance: Core
● Key Topics: neurodiversity, individual_differences, assessment_critique, 

biological_markers, data_analysis_methods, subgrouping, normative_modeling
● Tags: #neurodiversity, #neuroscience, #individualdifferences, #brainmapping, 

#subgrouping, #methodology, #GABA, #standardizedtestingcritique, 
#normativemodeling, #dimensionalapproach

CRITICAL QUOTES FOR ENLITENS

"Imaging transcriptomics, the study of correlations between gene-expression patterns and 
spatially varying properties of brain structure and function', has become a powerful tool for 
exploring the putative molecular underpinnings of neurotypical and neurodiverse brain 
organization."

"imaging transcriptomics also holds promise for dissecting the cortical architecture of complex 
neurotransmission systems and neuromodulatory pathways."

"While there is often no 1:1 relationship between gene expression and receptor density, 
evidence suggests that patterns of regional variations in gene expression can provide important 
insights into the functional role of a molecular target (e.g10,11.)."

"This implies that the cortical expression profiles of specific molecular targets may reflect their 
functional significance, which could be investigated by spatially aligning (i.e., correlating) IDPs 
with candidate gene expression patterns."

"However, linking spatially-dense gene expression patterns to highly variable IDPs, both across 
individuals and brain regions, is a computational and statistical challenge."

"Our findings indicate that the cortical transcriptomic landscape of genes encoding for specific 
pharmacological targets may be indicative of their clinical or behavioral relevance, and so guide 
the development of targeted pharmacotherapies in the future."

"Notably, these large-scale canonical expression patterns of modules not only align with the 
diverse spatial scales and temporal epochs of human brain organization-ranging from 
cytoarchitectonic boundaries to markers of neuronal subtypes-but also seems to be functionally 
relevant."

"This is particularly relevant when examining IDPs in neuropsychiatric and neurodevelopmental 
conditions that are marked by highly diverse and individualized neuroanatomical and functional 
variations in the brain (e.g..)."

"As the developmental trajectory of CT has an inverted U-shape across the human life span, 
positive deviations from the typical trajectory of CT are commensurate with delayed brain 
maturation."

"A gene with persistently high expression levels across the cortical surface may therefore exhibit
only a weak correlation with a regionally highly variable IDP, yet still have a significant impact on
a phenotype. Thus, the impact of a gene on a phenotype cannot be inferred solely based on 
spatial correlation."

"These studies show that a person's neuroanatomy is marked by highly individualized patterns 
of neuroanatomical variability, which may serve as a distinct neuroanatomical fingerprint that 
may be utilized for stratification purposes"

KEY STATISTICS & EVIDENCE

● Subgrouping based on Neuroanatomy: The study cohort was divided into two distinct 

subgroups based on their neuroanatomical profiles. "Subgroup 1 consisted of 178 
individuals (65 females, 113 males)... Subgroup 2 consisting of 101 individuals (36 
females, 65 males) and was characterized by positive correlations with the cortical 
Cluster 2, and by negative correlations with the limbic GABAA subunit Cluster 1."
● Behavioral Differences Between Subgroups: The neuroanatomically-defined 

subgroups showed significant differences in self-reported anxiety and depression. 
"Individuals in Subgroup 1 exhibited significantly elevated self-reported levels of anxiety 
(t(74)=2.52, p<0.01, padj=0.03, one-tailed) and depression (t(73)=3.65, p<0.001, 
padj<0.01 one-tailed) compared to those in Subgroup 2 (Fig. 5b)."

● Brain-Behavior Correlations: "In adults, we observed a significant positive correlation 

between levels of anxiety and neuroanatomical diversity within the limbic GABAA 
subunit Cluster 1 mask, which contained subunits α2,3.5,β1-3, ε, and y₁ (r=0.26 
t(80)=2.41, p<0.01 one-tailed)." "Here, as predicted from the subgroup analyses above 

(see Fig. 5b), more positive deflections from the typical CT trajectory were associated 
with elevated self-reported levels of anxiety (Fig. 5c, left panel)."

● Specificity of Correlation: "This relationship was absent within the mask representing 
the more unspecific (i.e., region-overarching) cortical expression pattern of GABAA 
subunit Cluster 2, which contained subunits α1,4, β2, Y2,3, and (r=0.005, t(80)=0.04, 
p=0.48) (Fig. 5c, right panel; see Supplementary Data Fig. 9 for model's generalization 
performance)."

● Age-Related Findings: "No significant differences in self-reported anxiety or depression
levels were observed between subgroups among adolescents, nor in parent-reported 
measures for children (Fig. 5b)." "No significant correlations were observed between 
variation in CT for levels of depression, and for anxiety/depression scores in children 
and adolescents (all p-values > 0.05)."

METHODOLOGY DESCRIPTIONS

Critique of Different Analytical Methods

The paper provides a detailed comparison of different high-level statistical techniques, 
highlighting that no single method is perfect and each has significant drawbacks—a powerful 
analogy for the limitations of standardized testing.

"Overall, the gradient-based approach provided the best trade-off between sensitivity and 
specificity across various levels of statistical stringency, identifying a reasonable number of 
significant genes suitable for downstream enrichment analysis (between 100 and 2000 at 
padj<0.001) (Fig. 3b, Supplementary Data Figs. 2, 3b). In comparison, LME-decoding identified 
the largest number of significant transcriptomic associations at padi<0.05. Yet, this number 
decreased rapidly when more conservative p-value thresholds were applied (Fig. 3b, 
Supplementary Data Figs. 2, 3b). Thus, although LME-decoding has substantial exploratory 
potential for detecting transcriptomic associations, it is also prone to generating false positives, 
likely due to spatial autocorrelations within the embedded transcriptomic maps. In contrast, 
GLS-decoding resulted in the lowest false positive rate (FPR) and yielded findings that were 
both sensitive and specific. However, incorporating the full spatial autoregressive correlation 
structure alongside stringent FDR adjustments may result in overly conservative findings, 
especially at more conservative levels of statistical stringency, where only a few significant 
genes were observed (Fig. 3d, Supplementary Data Figs. 2, 3d). Thus, while GLS-decoding 
seems well suited for hypothesis and enrichment testing, it is less optimal for broader 
exploratory analyses."

Normative Modeling: An Alternative to a Single Standard

The study’s method for analyzing individual brain scans rejects a simple "normal vs. abnormal" 
dichotomy. Instead, it places each person on a continuum of typical development to understand 
their unique neuroanatomical variations, which is perfectly aligned with Enlitens' strengths-
based, dimensional philosophy.

"To make individuals comparable, IDPs were standardized within the normative (i.e., 
neurotypical) range to account for the effects of age, sex, full-scale IQ (FSIQ), and other 
measures affecting brain structure (see Methods for details). Hence, instead of analyzing 

absolute CT metrics, all datasets were standardized relative to the canonical trajectory of brain 
development (Fig. 5a)."

"To make individuals comparable, IDPs were initially standardized within the neurotypical (i.e., 
non-ID) range by means of a General Linear Model (GLM) that included age, sex, FSIQ, 
acquisition site, and total brain volume as predictors. The model coefficients were subsequently 
used to predict CT across the cortex for all individuals in our sample, and the resulting residuals 
were centered and scaled. Thus, instead of employing absolute CT metrics, all datasets were 
normalized to unit standard deviations relative to the canonical developmental trajectory. Here, 
positive values indicated increased CT relative to the expected neurotypical range, while 
negative values indicated decreased CT. This approach was motivated by so-called normative 
modeling frameworks, which place each individual within a normative range of expected 
neurotypical variation. These studies show that a person's neuroanatomy is marked by highly 
individualized patterns of neuroanatomical variability, which may serve as a distinct 
neuroanatomical fingerprint that may be utilized for stratification purposes"

Stratifying Individuals Using Hierarchical Clustering

This section details the data-driven method used to group individuals based on their unique 
brain patterns, providing a practical example of a dimensional and non-categorical approach to 
understanding human diversity.

"Using hierarchical clustering, IDPs were then stratified according to their spatial similarity (i.e., 
neuroanatomical affinity) with GABAA subunit classes."

"Across multiple validity indices (see Methods), we discerned an optimal bifurcated clustering 
solution with a mean bootstrapped Jaccard similarity index of 0.714 for the primary cluster, and 
of 0.591 for the secondary cluster. Accordingly, our cohort was divided into two 
neuroanatomically distinct subgroups, each showing a different neuroanatomical association 
with the limbic and cortical expression signatures of GABAA subunit Clusters 1 and 2 (Fig. 5a)."

"To stratify individuals based on their spatial alignment with GABAA receptor subunit genes, the 
matrix of spatial correlations was then subjected to hierarchical clustering as outlined above, 
i.e., using NbClust to identify the optimal number of clusters, and clusterboot to establish their 
stability. Notably, this approach diverged from using a correlation matrix as input to the 
clustering algorithm but instead identified consistent patterns of high/low spatial correlations 
across individuals."

THEORETICAL FRAMEWORKS

The entire study is built on the framework of "imaging transcriptomics," a cutting-edge field that 
directly supports the idea that our observable brain features are linked to underlying biological 
diversity. This provides a scientific foundation for rejecting simplistic, purely behavioral labels.

"Imaging transcriptomics has become a power tool for linking imaging-derived phenotypes 
(IDPs) to genomic mechanisms. Yet, its potential for guiding CNS drug discovery remains 
underexplored. Here, utilizing spatially-dense representations of the human brain transcriptome,
we present an analytical framework for the transcriptomic decoding of high-resolution surface-

based neuroimaging patterns, and for linking IDPs to the transcriptomic landscape of complex 
neurotransmission systems in vivo."

"Imaging transcriptomics, the study of correlations between gene-expression patterns and 
spatially varying properties of brain structure and function', has become a powerful tool for 
exploring the putative molecular underpinnings of neurotypical and neurodiverse brain 
organization. Here, large open-access repositories featuring genome-wide expression profiles 
sampled across the brain, e.g., the Allen Human Brain Atlas (AHBA²), are used to identify genes
with an expression signature that spatially aligns with a structural or functional imaging 
phenotype."

POPULATION-SPECIFIC FINDINGS

Differences in Findings Between Adults and Children/Adolescents

This section is crucial evidence for Enlitens' argument that assessment tools are not universally 
applicable. The paper explicitly states that its findings hold for adults but not younger 
populations and discusses why, highlighting that the very data used to build the model (adult 
brain atlases) makes it less applicable to other groups. This is a direct parallel to how 
standardized tests built on one demographic fail others.

"Individuals in Subgroup 1 displayed a pattern of CT variability that positively correlated with the 
limbic GABAA subunit Cluster 1. IDPs of individuals in Subgroup 2 were positively correlated 
with the co-expression signatures of cortically-expressed GABAA subunit genes in Cluster 2. As
the developmental trajectory of CT has an inverted U-shape across the human life span, 
positive deviations from the typical trajectory of CT are commensurate with delayed brain 
maturation. In line with this, individuals - and adults in particular with more atypical CT in the 
limbic brain circuitry, which was characterized by high expression of the α2- containing GABAA 
subunit Cluster 1, also had significantly higher levels of anxiety and depression than adults 
falling into the α1-containing GABAA subunit Cluster 2. Several factors could explain why these 
correlations were observed in adults but not in children and adolescents. One possibility is a 
discrepancy between self-reported and parent-reported levels of depression and anxiety, which 
appears to diminish with age. Additionally, both the AHBA and PET atlas data are derived from 
adult samples. Thus, transcriptomic associations may be more accurate in adult populations 
compared to younger age groups." 

