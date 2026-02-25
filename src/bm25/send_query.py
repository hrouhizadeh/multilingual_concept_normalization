import argparse
import logging
import pandas as pd
from pathlib import Path
from elasticsearch import Elasticsearch
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BM25Searcher:
    def __init__(self, url: str):
        self.es = Elasticsearch([url], retry_on_timeout=True, timeout=100)
        if not self.es.ping():
            raise ConnectionError(f"Could not connect to Elasticsearch at {url}")

    def search(self, query_text: str):
        body = {
            "query": {"bool": {"should": {"match": {Config.INPUT_FIELD: str(query_text)}}}},
            "fields": ["_id"],
            "_source": False
        }
        res = self.es.search(index=Config.INDEX_NAME, body=body, size=Config.TOP_K)
        hits = res.get('hits', {}).get('hits', [])
        return [h['_id'] for h in hits], [h['_score'] for h in hits]

    def run_pipeline(self, input_dir: str, output_dir: str):
        input_path, save_path = Config.setup_paths(input_dir, output_dir)
        files = sorted(list(input_path.glob("*.csv")))
        
        for csv_file in files:
            logger.info(f"Processing: {csv_file.name}")
            df = pd.read_csv(csv_file)

            ids_col, scores_col = [], []
            for i, term in enumerate(df['term']):
                ids, scores = self.search(term)
                ids_col.append(ids)
                scores_col.append(scores)
                if (i + 1) % 100 == 0:
                    logger.info(f"Progress: {i + 1}/{len(df)}")

            df['retrieved_cuis'] = ids_col
            df['retrieved_scores'] = scores_col
            
            output_file = save_path / f"{csv_file.stem}_bm25.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BM25 Search Pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV directory")
    parser.add_argument("--output", required=True, help="Path to output base directory")
    args = parser.parse_args()

    searcher = BM25Searcher(Config.ELASTIC_URL)
    searcher.run_pipeline(args.input, args.output)