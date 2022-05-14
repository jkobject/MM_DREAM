import sys
from owkin_mm_dream import main, preprocess
import pandas as pd

"""Entry point for owkin_mm_dream."""
if __name__ == "__main__":  # pragma: no cover
    # if sending two more arguments, running predict_more
    if len(sys.argv) == 5:
        clf = main(sys.argv[1], sys.argv[2], showcase=False)
        rna_data = pd.read_csv(sys.argv[3], sep="\t", index_col=0)
        clinical_data = pd.read_csv(sys.argv[4], sep=",").set_index("Patient")
        rna_data, unc_clinical_data = preprocess(rna_data, clinical_data)
        res = clf.predict(clf, rna_data, clinical_data)
        res.to_csv("new_predictions.csv")
    else:
        main(sys.argv[1], sys.argv[2])
