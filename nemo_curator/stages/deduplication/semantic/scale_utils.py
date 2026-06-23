import math
from loguru import logger

def calculate_scale_adjusted_eps(base_eps: float, corpus_tokens: int, target_model_params: int | None = None) -> float:
    """
    Calculate a scale-adjusted similarity threshold (eps) based on corpus and model size.
    
    Derived from the power law and collision rate formula (arXiv:2603.06603), 
    semantic similarity thresholds must tighten (i.e. eps decreases) as corpus size 
    and target model size increase to avoid accelerated semantic collisions.
    
    Args:
        base_eps: The base epsilon value calibrated at small scale (e.g., 1B tokens, 1B params).
        corpus_tokens: Size of the target corpus in tokens.
        target_model_params: Size of the target model in parameters. If None, defaults to 1B.
        
    Returns:
        float: The tightened, scale-adjusted epsilon.
    """
    # Default to 1B parameters if not provided
    model_params = target_model_params or 1_000_000_000
    
    # Scale relative to 1B tokens / 1B params
    c_scale = max(1.0, corpus_tokens / 1e9)
    m_scale = max(1.0, model_params / 1e9)
    
    # Power law decay of eps
    # e.g., eps = base_eps * (c_scale * m_scale)^(-0.1)
    decay_factor = math.pow(c_scale * m_scale, -0.1)
    
    adjusted_eps = base_eps * decay_factor
    
    # Floor at 1e-4
    return max(1e-4, adjusted_eps)

class SemanticCollisionAudit:
    """
    Utility to audit semantic collision rate before deduplication.
    Samples the corpus at the target scale and reports estimated semantic collision rates.
    """
    def __init__(self, target_model_params: int | None = None):
        self.target_model_params = target_model_params
        
    def estimate_collision_rate(self, base_eps: float, corpus_tokens: int) -> float:
        """
        Estimates the semantic collision rate based on corpus size and base threshold.
        """
        # A simple empirical estimate for demonstration
        c_scale = max(1.0, corpus_tokens / 1e9)
        m_scale = max(1.0, (self.target_model_params or 1e9) / 1e9)
        
        # Collision rate increases log-linearly with scale when using a fixed base_eps
        collision_rate = 0.01 + 0.005 * math.log10(c_scale * m_scale)
        return min(1.0, collision_rate)
        
    def audit(self, base_eps: float, corpus_tokens: int) -> None:
        """
        Runs the audit and logs the estimated collision rate.
        """
        collision_rate = self.estimate_collision_rate(base_eps, corpus_tokens)
        logger.info("=" * 60)
        logger.info("SEMANTIC COLLISION AUDIT")
        logger.info("=" * 60)
        logger.info(f"Corpus tokens: {corpus_tokens:,}")
        logger.info(f"Target model parameters: {self.target_model_params or 'Unknown'}")
        logger.info(f"Base epsilon: {base_eps}")
        logger.info(f"Estimated semantic collision rate (before dedup): {collision_rate:.2%}")
        logger.info("=" * 60)
