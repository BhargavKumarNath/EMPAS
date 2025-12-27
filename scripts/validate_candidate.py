import sys
import os
import logging
import hydra 
from omegaconf import DictConfig
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.validator import Validator
from src.core.search_space import Genome

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

BEST_GENOME = [4, 16, 8, 8, 4, 4, 4, 8, 8, 8, 8, 4, 4, 8, 8, 8, 16, 16, 4, 8, 16, 16]

# Baseline (All FP16) for comparison
BASELINE_GENOME = [16] * 22

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    logger.info("=== EMPAS: Final Validation ===")

    # 1. Setup Validator (Loads Model)
    validator = Validator("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # 2. Validator Baseline
    logger.info("\n--- Validating FP16 Baseline ---")
    validator.apply_genome(Genome(genes=BASELINE_GENOME))
    base_loss, base_lat = validator.validate()
    logger.info(f"Baseline -> Loss: {base_loss:.4f}, Latency: {base_lat:.2f}ms")

    # 3. Reload Model
    del validator
    validator = Validator("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # 4. Validate Found Solution
    logger.info("\n--- Validating Evolved Architecture ---")
    validator.apply_genome(Genome(genes=BEST_GENOME))
    evolved_loss, evolved_lat = validator.validate()
    logger.info(f"Evolved  -> Loss: {evolved_loss:.4f}, Latency: {evolved_lat:.2f}ms")

    # 5. Analysis
    logger.info("\n=== RESULTS ===")
    logger.info(f"Accuracy Drop: {evolved_loss - base_loss:.4f} (Target: Small)")

    # Calculate Theoretical Compression
    total_bits = sum(BEST_GENOME)
    max_bits = 16 * 22
    compression = 1.0 - (total_bits / max_bits)
    logger.info(f"Model Compression: {compression*100:.1f}%")

if __name__ == "__main__":
    main()