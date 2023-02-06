import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import io

from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support, plot_roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sksurv.nonparametric import kaplan_meier_estimator

import synapseclient
from biomart import BiomartServer

CLINICAL_LABELS = {
    "D_OS": "survival_days",
    "D_OS_FLAG": "is_deceased",
    "D_PFS": "pfs_days",
    "D_PFS_FLAG": "has_progressed",
    "HR_FLAG": "to_pred",
}
CLINICAL_FEATURES = {"D_Age": "age", "D_Gender": "sex", "D_ISS": "stage"}
MM_PATHWAY = os.path.dirname(os.path.abspath(__file__)) + "/../data/MMpathways.gmt"

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


def load_data(
    syn_login,
    syn_password,
    rna_data="syn9744875",
    clinical_data="syn9926878",
    explanation_table="syn9744732",
):
    """Loads and processes the data.
    
    Args:
        syn_login (str): Synapse login.
        syn_password (str): Synapse password.
        rna_data (str, optional): RNAseq data. Defaults to "syn9744875".
        clinical_data (str, optional): Clinical data. Defaults to "syn9926878".
        explanation_table (str, optional): Explanation table. Defaults to "syn9744732".

    Returns:
        rna_data (pandas.DataFrame): RNAseq data.
        clinical_data (pandas.DataFrame): Clinical data.
    """
    syn = synapseclient.Synapse()
    syn.login(syn_login, syn_password)

    # Obtain a pointer and download the data
    # loading the DREAM Challenge MM RNAseq dataset from Synapse
    # dataset analysed with Salmon (reference-free mapper),
    # using hg19 gene annotation annotated with ENTREZ id + ERCC (likely ERCC92) which is the spike-in gene which are not present here
    rna_data = syn.get(entity=rna_data)
    explanation_table = syn.get(entity=explanation_table)
    clinical_data = syn.get(entity=clinical_data)

    # loading the RNAseq data from Synapse
    rna_data = pd.read_csv(rna_data.path, sep="\t", index_col=0)
    clinical_data = pd.read_csv(clinical_data.path, sep=",").set_index("Patient")
    explanation_table = pd.read_csv(explanation_table.path, sep=",", index_col=0)

    # showing the explanation_table of the data
    print("\nhere is our clinical data: ")
    print(
        explanation_table.loc[set(clinical_data.columns) & set(explanation_table.index)]
    )
    return rna_data, clinical_data


def preprocess(rna_data, clinical_data):
    print("the rna data has {} samples".format(len(rna_data.columns)))
    # subsetting columns
    clinical_data = clinical_data.rename(
        columns={**CLINICAL_LABELS, **CLINICAL_FEATURES}
    )
    unc_clinical_data = clinical_data[clinical_data.to_pred != "CENSORED"]
    unc_clinical_data = unc_clinical_data[
        list(CLINICAL_FEATURES.values()) + list(CLINICAL_LABELS.values())
    ]
    unc_clinical_data = unc_clinical_data[~unc_clinical_data["stage"].isna()]

    # getting GENE symbols
    gene_names = generateGeneNames()
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
    return rna_data, unc_clinical_data


def make_dataset(rna_data, clinical_data, pathway_file=MM_PATHWAY):
    """Makes a dataset from the RNAseq data and clinical data.
    
    Args:
        rna_data (pandas.DataFrame): RNAseq data.
        clinical_data (pandas.DataFrame): Clinical data.
        pathway_file (str, optional): Pathway file. Defaults to "../data/mm_pathway.txt".

    Returns:
        pandas.DataFrame: Dataset.
    """
    # sorting to get the latest rnaseq available for a patient. By postulating that the number is equal to the timepoint at sampling
    col = list(rna_data.columns)
    col.sort()
    rna_data = rna_data[col]

    # removing duplicates and aassociating a single profile with a single patient
    rna_data.columns = ["_".join(i.split("_")[:2]) for i in rna_data.columns]
    for val in dups(rna_data.columns):
        loc = rna_data[val].iloc[:, -1]
        rna_data.drop(val, axis=1, inplace=True)
        rna_data[val] = loc.values

    to_add = (
        clinical_data[CLINICAL_FEATURES.values()]
        .replace({"Female": 1, "Male": 0})
        .astype(int)
    )

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
    return X


def main(
    syn_login,
    syn_password,
    predict_on=DEFAULT_FEATURES,
    pathway_file=MM_PATHWAY,
    showcase=True,
):
    """Main entry point for owkin_mm_dream.

    The main function executes on commands:
    `python -m owkin_mm_dream` and `$ owkin_mm_dream `.
    
    Args:
        syn_login (str): Synapse login.
        syn_password (str): Synapse password.
        predict_on (list): List of features to predict on.
        pathway_file (str): Path to pathway file.
    """

    rna_data, clinical_data = load_data(syn_login, syn_password)

    rna_data, unc_clinical_data = preprocess(rna_data, clinical_data)

    X = make_dataset(rna_data, unc_clinical_data, pathway_file)

    # generating the output data
    Y = (
        unc_clinical_data.replace({"FALSE": 0, "TRUE": 1})["to_pred"]
        .loc[X.index]
        .values
    )
    # scaling the data
    scaler = preprocessing.StandardScaler().fit(X[predict_on].values)
    X_scaled = scaler.transform(X[predict_on].values)

    if showcase:
        show_case(X_scaled, Y, predict_on=predict_on, clinical_data=unc_clinical_data)

    print("choosing logistic regression as best clf")

    clf = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=1, max_iter=800, C=0.2
    )
    clf = clf.fit(X_scaled, Y)
    return clf


def show_case(X_scaled, Y, predict_on=DEFAULT_FEATURES, clinical_data=None):
    # finding the best predictors
    clf = LogisticRegression()
    clf = clf.fit(X_scaled, Y)

    best_pred = np.array(predict_on)[abs(clf.coef_[0]) > 0.2]
    print("\nThe best predictors according to logistic regression are: ")
    print(best_pred)

    # showing a few models' predictions
    print("\n\nKNeighborsClassifier:\n")
    clf = KNeighborsClassifier(n_neighbors=15)
    clf = clf.fit(X_scaled, Y)
    show_metrics(clf, X_scaled, Y, clinical_data=clinical_data)
    print(
        "a KNN with 15 neigbhors shows a lower f1 score overall (CV=100) but a really good AUC. \
However, this is likely due to overfitting."
    )

    print("_______________________________________\n\n")
    print("svm:\n")
    clf = svm.SVC(C=0.9)
    clf = clf.fit(X_scaled, Y)
    show_metrics(clf, X_scaled, Y, clinical_data=clinical_data)
    print(
        "\nSVM is able to get a very good precision and in medical applications, \
this is often what we are looking for: FPs need to be as low as possible. \
However, the kaplan-meyer curve is worrying. But it might be due to it \
being based on survival data whereas we are looking at risk / fast progression?"
    )

    print("_______________________________________\n\n")
    print("LogisticRegression:\n")
    clf = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=1, max_iter=800, C=0.2
    )
    clf = clf.fit(X_scaled, Y)
    show_metrics(clf, X_scaled, Y, clinical_data=clinical_data)
    print(
        "logistic regression with elasticnet and a high l1_ratio \
shows a good f1 score. Lower precision, But by far the best K-M curve of all."
    )


def show_metrics(
    clf, X, Y, clinical_data=None, time="survival_days", time_flag="is_deceased"
):
    """Shows the metrics of a model.

    Args:
        clf (object): A model.
        X (np.array): The input data.
        Y (np.array): The output data.
        clinical_data (pd.DataFrame): The clinical data.
        time (str): The time to predict on.
        time_flag (str): The time flag.
        
    """
    Y_pred = clf.predict(X)
    prec, rec, f1, _ = precision_recall_fscore_support(Y, Y_pred, average="weighted")
    plot_roc_curve(clf, X, Y)
    plt.show()
    cross_scores = cross_val_score(clf, X, Y, cv=100)

    # showing the kaplan-meier survival curve
    if clinical_data is not None:
        for pred_event in (1, 0):
            mask_treat = Y_pred == pred_event
            time_treatment, survival_prob_treatment = kaplan_meier_estimator(
                clinical_data[time_flag][mask_treat].astype(bool),
                clinical_data[time][mask_treat],
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
        plt.show()
    print("precision, recall, f1_score:")
    print(prec, rec, f1)
    print("k-fold CV score:")
    print(cross_scores.mean())


## copied from genepy


def _fetchFromServer(ensemble_server, attributes):
    server = BiomartServer(ensemble_server)
    ensmbl = server.datasets["hsapiens_gene_ensembl"]
    res = pd.read_csv(
        io.StringIO(
            ensmbl.search({"attributes": attributes}, header=1).content.decode()
        ),
        sep="\t",
    )
    return res


def createFoldersFor(filepath):
    """
  will recursively create folders if needed until having all the folders required to save the file in this filepath
  """
    prevval = ""
    for val in os.path.expanduser(filepath).split("/")[:-1]:
        prevval += val + "/"
        if not os.path.exists(prevval):
            os.mkdir(prevval)


def generateGeneNames(
    ensemble_server="http://nov2020.archive.ensembl.org/biomart",
    useCache=False,
    cache_folder="/".join(__file__.split("/")[:-3]) + "/",
    attributes=[],
):
    """generate a genelist dataframe from ensembl's biomart

  Args:
      ensemble_server ([type], optional): [description]. Defaults to ENSEMBL_SERVER_V.
      useCache (bool, optional): [description]. Defaults to False.
      cache_folder ([type], optional): [description]. Defaults to CACHE_PATH.

  Raises:
      ValueError: [description]

  Returns:
      [type]: [description]
  """
    attr = [
        "ensembl_gene_id",
        "clone_based_ensembl_gene",
        "hgnc_symbol",
        "gene_biotype",
        "entrezgene_id",
    ]
    assert cache_folder[-1] == "/"

    cache_folder = os.path.expanduser(cache_folder)
    createFoldersFor(cache_folder)
    cachefile = os.path.join(cache_folder, ".biomart.csv")
    if useCache & os.path.isfile(cachefile):
        print("fetching gene names from biomart cache")
        res = pd.read_csv(cachefile)
    else:
        print("downloading gene names from biomart")
        res = _fetchFromServer(ensemble_server, attr + attributes)
        res.to_csv(cachefile, index=False)

    res.columns = attr + attributes
    if type(res) is not type(pd.DataFrame()):
        raise ValueError("should be a dataframe")
    res = res[~(res["clone_based_ensembl_gene"].isna() & res["hgnc_symbol"].isna())]
    res.loc[res[res.hgnc_symbol.isna()].index, "hgnc_symbol"] = res[
        res.hgnc_symbol.isna()
    ]["clone_based_ensembl_gene"]

    return res


def dups(lst):
    """
        shows the duplicates in a list
    """
    seen = set()
    # adds all elements it doesn't know yet to seen and all other to seen_twice
    seen_twice = set(x for x in lst if x in seen or seen.add(x))
    # turn the set into a list (as requested)
    return list(seen_twice)

