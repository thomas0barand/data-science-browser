"""Run complete training and evaluation pipeline"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import setup_logging
from src.data_loader import DataLoader
from src.eda import EDA
from src.models.embedding_model import EmbeddingModel
from src.training import Trainer
from src.evaluation import Evaluator
from configs.config import DATA_DIR, FIGURES_DIR

logger = setup_logging()


def main():
    logger.info("Starting pipeline...")
    
    # Load data
    data_loader = DataLoader(DATA_DIR)
    data_loader.load_all()
    
    # EDA
    eda = EDA(data_loader.corpus, data_loader.queries, data_loader.validation)
    eda.basic_statistics()
    eda.analyze_text_lengths()
    eda.plot_distributions(FIGURES_DIR)
    
    # Train model
    model = EmbeddingModel()
    trainer = Trainer(model)
    trainer.train(data_loader.corpus, data_loader.queries, data_loader.validation)
    
    # Evaluate
    evaluator = Evaluator(model)
    results = evaluator.evaluate(data_loader.validation)
    
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()

