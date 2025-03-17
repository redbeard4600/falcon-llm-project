# monitoring.py
import time
import psutil
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class InferenceStats:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    inference_time_ms: float
    memory_usage_mb: float
    timestamp: datetime = datetime.now()

class ModelMonitor:
    def __init__(self):
        self.inference_history: List[InferenceStats] = []
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='model_monitoring.log'
        )
        self.logger = logging.getLogger('model_monitor')
    
    def track_inference(self, tokenizer, prompt: str, completion: str) -> InferenceStats:
        """Track a single inference operation"""
        start_time = time.time()
        prompt_tokens = len(tokenizer.encode(prompt))
        process = psutil.Process()
        
        # Get memory before inference
        memory_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        
        # Here you would normally do the actual inference
        # For tracking purposes, we'll just measure time and memory
        
        memory_after = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        end_time = time.time()
        
        completion_tokens = len(tokenizer.encode(completion))
        
        stats = InferenceStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            inference_time_ms=(end_time - start_time) * 1000,
            memory_usage_mb=memory_after - memory_before
        )
        
        self.inference_history.append(stats)
        self.logger.info(f"Inference completed: {stats}")
        return stats
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics from all tracked inferences"""
        if not self.inference_history:
            return {}
        
        total_inferences = len(self.inference_history)
        avg_time = sum(stat.inference_time_ms for stat in self.inference_history) / total_inferences
        avg_tokens = sum(stat.total_tokens for stat in self.inference_history) / total_inferences
        max_time = max(stat.inference_time_ms for stat in self.inference_history)
        
        return {
            "total_inferences": total_inferences,
            "avg_inference_time_ms": avg_time,
            "avg_tokens_per_inference": avg_tokens,
            "max_inference_time_ms": max_time,
            "total_tokens_processed": sum(stat.total_tokens for stat in self.inference_history)
        }

# Create a singleton instance
monitor = ModelMonitor()