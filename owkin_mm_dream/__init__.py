import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support, plot_roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sksurv.nonparametric import kaplan_meier_estimator

import synapseclient


CLINICAL_LABELS = {
    "D_OS": "survival_days",
    "D_OS_FLAG": "has_survived",
    "D_PFS": "pfs_days",
    "D_PFS_FLAG": "has_progressed",
    "HR_FLAG": "to_pred",
}
CLINICAL_FEATURES = {"D_Age": "age", "D_Gender": "sex", "D_ISS": "stage"}
MM_PATHWAY = "../data/mm_pathway.txt"

DEFAULT_FEATURES = [
    "age",
    "stage",
    "PHF19",
    "BCL2A1",
    "BIOCARTA_DEATH_PATHWAY",
    "BIOCARTA_CELLCYCLE_PATHWAY",
    "BIOCARTA_ERK_PATHWAY",
    "BIOCARTA_NFKB_PATHWAY",
    "BIOCARTA_P53_PATHWAY",
    "BIOCARTA_RAS_PATHWAY",
    "BORTEZOMIB_ABNORMALITY_OF_CHROMOSOME_STABILITY",
    "REACTOME_ANTIGEN_PROCESSING_UBIQUITINATION_PROTEASOME_DEGRADATION",
    "REACTOME_DNA_REPAIR",
    "UAMS17",
    "THALIDOMIDE",
]


def main(syn_login, syn_password, predict_on=DEFAULT_FEATURES, pathway_file=MM_PATHWAY):
    """Main entry point for owkin_mm_dream.
    
    Args:
        syn_login (str): Synapse login.
        syn_password (str): Synapse password.
        predict_on (list): List of features to predict on.
        pathway_file (str): Path to pathway file.

    Returns:
    """

    syn = synapseclient.Synapse()
    syn.login(syn_login, syn_password)

    # Obtain a pointer and download the data
    # loading the DREAM Challenge MM RNAseq dataset from Synapse
    # dataset analysed with Salmon (reference-free mapper),
    # using hg19 gene annotation annotated with ENTREZ id + ERCC (likely ERCC92) which is the spike-in gene which are not present here
    syn9744875 = syn.get(entity="syn9744875")
    syn9744732 = syn.get(entity="syn9744732")
    syn9926878 = syn.get(entity="syn9926878")
    # loading the RNAseq data from Synapse
    rna_data = pd.read_csv(syn9744875.path, sep="\t", index_col=0)
    clinical_data = pd.read_csv(syn9926878.path, sep=",").set_index("Patient")
    explanation = pd.read_csv(syn9744732.path, sep=",", index_col=0)

    # showing the explanation of the data
    explanation.loc[set(clinical_data.columns) & set(explanation.index)]

    # subsetting columns
    clinical_data = clinical_data.rename(
        columns={**CLINICAL_LABELS, **CLINICAL_FEATURES}
    )
    unc_clinical_data = clinical_data[clinical_data.to_pred != "CENSORED"]
    unc_clinical_data = unc_clinical_data[
        list(CLINICAL_FEATURES.values()) + list(CLINICAL_LABELS.values())
    ]
    unc_clinical_data = unc_clinical_data[~unc_clinical_data["stage"].isna()]

    to_add = (
        unc_clinical_data[CLINICAL_FEATURES.values()]
        .replace({"Female": 1, "Male": 0})
        .astype(int)
    )
    # getting GENE symbols
    gene_names = ghelp.generateGeneNames()
    gene_names = gene_names[gene_names.gene_biotype == "protein_coding"].set_index(
        "ensembl_gene_id"
    )

    # subset to coding genes
    rna_data = rna_data.loc[set(gene_names.index) & set(rna_data.index)]
    rna_data[rna_data.var(1) != 0]
    rna_data.index = [
        gene_names.loc[val]["hgnc_symbol"]
        if type(gene_names.loc[val]["hgnc_symbol"]) is str
        else gene_names.loc[val].iloc[0]["hgnc_symbol"]
        for val in rna_data.index
    ]

    # sorting to get the latest rnaseq available for a patient. By postulating that the number is equal to the timepoint at sampling
    col = list(rna_data.columns)
    col.sort()
    rna_data = rna_data[col]

    # removing duplicates and aassociating a single profile with a single patient
    rna_data.columns = ["_".join(i.split("_")[:2]) for i in rna_data.columns]
    for val in ghelp.dups(rna_data.columns):
        loc = rna_data[val].iloc[:, -1]
        rna_data.drop(val, axis=1, inplace=True)
        rna_data[val] = loc.values

    # generating the input data
    X = rna_data.T.loc[list(set(clinical_data.index))]
    X = pd.concat([X, to_add], axis=1)
    X = X[~X.age.isna()]

    # generating pathway level expression profiles
    # (a very coarse GSEA that works as well as true ssGSEA)
    pathways = {}
    with open(pathway_file) as f:
        for val in f.read().splitlines():
            pathways[val.split("\t")[0]] = val.split("\t")[2:]
    xcol = set(X.columns)
    print("\nWe are using these pathways & genesets: ")
    print(list(pathways.keys()))
    for k, val in pathways.items():
        X[k] = X[set(val) & xcol].sum(axis=1)

    # generating the output data
    Y = clinical_data.replace({"FALSE": 0, "TRUE": 1})["to_pred"].loc[X.index].values

    # scaling the data
    scaler = preprocessing.StandardScaler().fit(X[predict_on].values)
    X_scaled = scaler.transform(X[predict_on].values)

    # finding the best predictors
    clf = LogisticRegression()
    clf = clf.fit(X_scaled, Y)

    best_pred = np.array(predict_on)[abs(clf.coef_[0]) > 0.2]
    print("\nThe best predictors according to logistic regression are: ")
    print(best_pred)

    # showing a few models' predictions
    print("")
    clf = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=1, max_iter=800, C=0.2
    )
    clf = clf.fit(X_scaled, Y)
    show_metrics(clf, X_scaled, Y)

    print("")
    clf = svm.SVC(C=0.9)
    clf = clf.fit(X_scaled, Y)
    show_metrics(clf, X_scaled, Y)

    print("")
    clf = KNeighborsClassifier(n_neighbors=15)
    clf = clf.fit(X_scaled, Y)
    show_metrics(clf, X_scaled, Y)


def show_metrics(clf, X_scaled, Y, clinical_data=None):
    Y_pred = clf.predict(X_scaled)
    prec, rec, f1, _ = precision_recall_fscore_support(Y, Y_pred, average="weighted")
    plot_roc_curve(clf, X_scaled, Y)
    plt.show()
    cross_scores = cross_val_score(clf, X_scaled, Y, cv=100)

    # showing the kaplan-meier survival curve
    if clinical_data is not None:
        for pred_event in (1, 0):
            mask_treat = Y_pred == pred_event
            time_treatment, survival_prob_treatment = kaplan_meier_estimator(
                clinical_data["D_OS_FLAG"][mask_treat],
                clinical_data["D_OS"][mask_treat],
            )

            plt.step(
                time_treatment,
                survival_prob_treatment,
                where="post",
                label="EVENT_PREDICTION = %s" % pred_event,
            )
        plt.ylabel("est. probability of survival $\hat{S}(t)$")
        plt.xlabel("time $t$")
        plt.legend(loc="best")
    print("precision, recall, f1_score:")
    print(prec, rec, f1)
    print("k-fold CV score:")
    print(cross_scores.mean())
