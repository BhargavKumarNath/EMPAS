import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple
from copy import deepcopy
from .wrapper import ModelWrapper
from .quantizer import fake_quantize_tensor
logger = logging.getLogger(__name__)

class SensitivityProfiler:
    def __init__(self, wrapper: ModelWrapper, dataset: List[torch.Tensor]):
        self.wrapper = wrapper
        self.dataset = dataset
        self.choices = [2, 4, 8]

    def get_baseline_loss(self) -> float:
        """Measures FP16 loss once"""
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.dataset:
                total_loss += self.wrapper.forward_pass_check(batch)
        return total_loss / len(self.dataset)
    
    def profile(self) -> Dict[Tuple[int, int], float]:
        """Returns a dictionary mapping (layer_idx, bit_width) -> Perplexity Increase"""
        results = {}

        # 1. Get Baseline
        logger.info("Computing baseline FP16 loss...")
        baseline_loss = self.get_baseline_loss()
        logger.info(f"Baseline Loss: {baseline_loss:.4f}")

        # 2. Iterate Layers
        layers = self.wrapper.model.model.layers

        for i, layer in enumerate(layers):
            logger.info(f"Profiling Layer {i}/{len(layers)-1}...")

            # Quantise all linear weights int he layer to the target bit width
            linear_modules = [m for m in layer.modules() if isinstance(m, nn.Linear)]

            # Backup original weights (CPU to save VRAM)
            original_weights = {m: m.weight.detach().cpu().clone() for m in linear_modules}

            for bits in self.choices:
                # A. Apply Fake Quantisation
                for m in linear_modules:
                    w = m.weight.data
                    m.weight.data = fake_quantize_tensor(w, bits)

                # B. Measure Loss
                total_loss = 0.0
                with torch.no_grad():
                    for batch in self.dataset:
                        total_loss += self.wrapper.forward_pass_check(batch)
                avg_loss = total_loss / len(self.dataset)

                # C. Record Sensitivity (Delta Loss)
                sensitivity = max(0.0, avg_loss - baseline_loss)
                results[(i, bits)] = sensitivity

                # D. Restore Weights for next bit-width
                for m in linear_modules:
                    m.weight.data = original_weights[m].to(self.wrapper.device)
            
            # Ensure weights are restored before moving to next layer
            for m in linear_modules:
                m.weight.data = original_weights[m].to(self.wrapper.device)
                
        return results, baseline_loss