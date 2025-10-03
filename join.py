import pandas as pd


def join_results(json_path, csv_path):
    json_df = pd.read_json(json_path)
    prior_metrics = ["length", "Precision", "Recall", "F1"]
    forbidden_metrics = [f"forbidden_{m}" for m in prior_metrics]
    required_metrics = [f"required_{m}" for m in prior_metrics]
    json_metrics = ['filename', *forbidden_metrics, *required_metrics]
    json_df = json_df[json_metrics]
    json_df = json_df.rename(columns={
        k: f"prior_{k}" for k in json_df.columns
    })

    csv_df = pd.read_csv(csv_path)
    merged = pd.merge(
        csv_df,
        json_df,
        left_on=["dataset"],
        right_on=["prior_filename"],
        how="left"
    )

    prior_cols = [col for col in merged.columns if col.startswith("prior_") and col != "prior_filename"]
    if "impl" in merged.columns:
        merged.loc[merged["impl"] != "new", prior_cols] = float('nan')

    return merged

if __name__ == "__main__":
    for type_ in ["bnlearn", "synthetic"]:
        json_path = f"{type_}.json"
        csv_path = f"{type_}-results.csv"
        merged_df = join_results(json_path, csv_path)
        output_path = f"merged_{type_}.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"Joined results saved to {output_path}")
