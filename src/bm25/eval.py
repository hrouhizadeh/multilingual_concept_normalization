import pandas as pd
import ast
import argparse
from pathlib import Path

def evaluate_results(folder_path: str, output_name: str):
    folder = Path(folder_path)
    summary_results = []

    for file_path in folder.glob("*.csv"):
        df = pd.read_csv(file_path)
        
        # Safe evaluation of list strings
        df['code'] = df['code'].apply(ast.literal_eval)
        df['retrieved_cuis'] = df['retrieved_cuis'].apply(ast.literal_eval)

        file_metrics = {'filename': file_path.name, 'number_of_rows': len(df)}

        for k in [1, 3, 5, 10]:
            hits = [
                1 if any(c in row['retrieved_cuis'][:k] for c in row['code']) else 0 
                for _, row in df.iterrows()
            ]
            file_metrics[f'R@{k}'] = (sum(hits) / len(hits)) * 100 if len(hits) > 0 else 0

        summary_results.append(file_metrics)

    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(output_name, index=False)
    print(f"Evaluation complete. Summary saved to {output_name}")
    print(summary_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BM25 Recall")
    parser.add_argument("--folder", required=True, help="Path to the folder containing result CSVs")
    parser.add_argument("--out", default="recall_summary.csv", help="Output summary filename")
    args = parser.parse_args()
    
    evaluate_results(args.folder, args.out)