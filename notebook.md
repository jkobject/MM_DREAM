# Owkin miniproject notebook:

## Monday

- google MM relapse litt.
- thinking about the plan

- thinking about the project's solution:


(2h)

## Tuesday

none

## Wednesday

(1h)

## Plan

- litt review
- setting up env
- playing with the data
- making a few metrics
- making it reproducible: docker / reqs / github
- writing down final info: readme / doc / linting

## Potential solutions

- reduce features:
  - dim reduction: (PCA / gene set enrichment with ssGSEA over pathways)
  - manual curation based on known litterature

- classify on the set of features
  - explainable: logistic regression, DT
  - complex: RF
  
- visualize:
  - use the survival curves
  - 


## litterature review

- [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8467450/](Multiple Myeloma Relapse Is Associated with Increased NFκB Pathway Activity and Upregulation of the Pro-Survival BCL-2 Protein BFL-1 (BCL2A1), BCL-XL)
-[https://ashpublications.org/bloodadvances/article/5/9/2391/475878/Comprehensive-CRISPR-Cas9-screens-identify-genetic](Genetic inactivation of DNA damage repair pathway regulators enhances sensitivity to cytotoxic chemotherapy.)
- [https://www.synapse.org/#!Synapse:syn7222203](MM dream challenge full dataset.)
- [https://www.nature.com/articles/s41375-020-0742-z](a simple four feature predictor composed of age, ISS, and expression of PHF19 and MMSET performs similarly to more complex models with many more gene expression features included.)

### Notes:

- relapse effect and will depend at least in part on the treatment used
- some of the determminants of relapse might not be seen in the pre-treatment expression profile of MM tumors. MM might react differently due to genomics nd epigenomics differences that would only manifest during MM's reaction to treatment.
- the full dataset actually contains SV with strelka, SNPs/Indels wwith M2, FISH-seq and lots more clinical annotions.
- In the initial dream challenge, RNAseq alone did better. 
- best metric is integrated AUC

- i) Chromosomal abnormalities [3]: deletion of chromosome 1p, gain of chromosome 1q, gain of chromosome 9, deletion of chromosome 13q, deletion of chromosome 17p, translocation t(4;14), trainslocation t(11;14), translocation t(14;16/14;20).
- ii) DNA repair pathways [5]: non-homologous end-joining pathway, homologous recombination pathway, Fanconi anemia pathway, nucleotide excision repair pathway, mismatch repair pathway, base excision repair pathway.
iii) Other pathways [3]: cell cycle pathway, p53 signalling pathway, NF-kB signalling pathway, Ras-ERK pathway.
- iv) Genes targeted by Multiple Myeloma treatments [6]: Bortezomib, Thalidomide
- v) Mutations associated with high-risk Multiple Myeloma [3].
- vi) Other gene expression profiles obtained from literature: EMC92 [7], UAMS70 [8], DNA repair pathway score [4], IFM group [9], cell death network [10].
- Sum of gene expressions were used, and feature selection of the engineered features was also performed to progressively discard uninformative features. We also include clinical data comprising age and ISS as features.


### Personal remarks:

- it has been shown that spike in is essential to show some gene expression change in some datasets.
- it has been shown that count expression data such as what is outputed by RSEM is actually better suited for statistical analysis than TPMs
- using the term "gender" to talk about a patient's sex is derogative.
- we are loosing 40% of our data due to missing class labels.
