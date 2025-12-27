import torch
import torch.nn as nn
import logging
import time
from typing import List, Dict, Any
from ..models.wrapper import ModelWrapper
from ..models.quantizer import fake_quantize_tensor
from ..utils.data import get_calibration_dataset
from .search_space import Genome
logger = logging.getLogger(__name__)

class Validator:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.wrapper = ModelWrapper(model_name, device)
        self.dataset = get_calibration_dataset(model_name, seq_len=2048, nsamples=8)
    
    def apply_genome(self, genome: Genome):
        """Aplies the mixed precision configuration to the loaded model"""
        layers = self.wrapper.model.model.layers
        if len(genome.genes) != len(layers):
            raise ValueError(f"Genome length {len(genome.genes)} != Model layers {len(layers)}")
        
        logger.info("Applying Mixed-Precision Quantization...")

        for i, (layer, bits) in enumerate(zip(layers, genome.genes)):
            linear_modules = [m for m in layer.modules() if isinstance(m, nn.Linear)]

            for m in linear_modules:
                with torch.no_grad():
                    w = m.weight.data
                    m.weight.data = fake_quantize_tensor(w, bits)
        logger.info(f"Model quantized. Configuration: {genome.genes}")
    
    def validate(self) -> float:
        """Run full evaluation on the test set"""
        logger.info("Starting Validation Loop...")
        self.wrapper.model.eval()
        total_loss = 0.0
        total_time = 0.0

        with torch.no_grad():
            for i, batch in enumerate(self.dataset):
                start = time.time()
                loss = self.wrapper.forward_pass_check(batch)
                end = time.time()

                total_loss += loss
                total_time += (end - start)
        avg_loss = total_time / len(self.dataset)
        avg_latency = (total_time / len(self.dataset)) * 1000

        return avg_loss, avg_latency
    