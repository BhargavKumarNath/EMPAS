import sys
import os
import torch
import json
import hydra
import logging
from omegaconf import DictConfig
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.data import get_calibration_dataset
from src.models.wrapper import ModelWrapper
from src.models.sensitivity import SensitivityProfiler
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

# Using TinyLlama for fast profiling validation
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    logger.info("=== EMPAS: Sensitivity Profiling ===")

    # 1. Load Resources
    dataset = get_calibration_dataset(MODEL_ID, seq_len=128, nsamples=4)
    wrapper = ModelWrapper(MODEL_ID, device="cuda")

    # 2. Run Profiler
    profiler = SensitivityProfiler(wrapper, dataset)
    sensitivity_table, baseline = profiler.profile()

    # 3. Save Results
    output = {
        "baseline_loss": baseline,
        "sensitivity": {f"{k[0]}_{k[1]}": v for k, v in sensitivity_table.items()}
    }

    os.makedirs("./data/profiles", exist_ok=True)
    save_path = "./data/profiles/tinyllama_sensitivity.json"

    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)
    
    logger.info(f"Profile saved to {save_path}")

    # 4. Sanity Check
    logger.info("\n--- Sanity Check: Layer 0 ---")
    print(f"2-bit Impact: {output['sensitivity']['0_2']:.4f}")
    print(f"4-bit Impact: {output['sensitivity']['0_4']:.4f}")
    print(f"8-bit Impact: {output['sensitivity']['0_8']:.4f}")

    if output['sensitivity']['0_2'] < output['sensitivity']['0_8']:
        logger.warning("WARNING: 2-bit appears better than 8-bit? Check Quantizer.")
    else:
        logger.info("SUCCESS: 2-bit degrades model more than 8-bit, as expected")

if __name__ == "__main__":
    main()

    