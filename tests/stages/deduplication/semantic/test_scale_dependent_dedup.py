import os
import random
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import make_blobs

with suppress(ImportError):
    from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow
    from nemo_curator.backends.xenna import XennaExecutor


@pytest.mark.gpu
class TestScaleDependentSemanticDeduplication:
    """Test that scale-dependent threshold adjustment works as expected."""

    def setup_method(self) -> None:
        """Setup method that creates synthetic data."""
        self.n_clusters = 5
        self.n_samples_per_cluster = [100 * (i + 1) for i in range(self.n_clusters)]
        self.n_features = 3

        random.seed(42)
        np.random.seed(42)  # noqa: NPY002
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.X, _ = make_blobs(
            n_samples=self.n_samples_per_cluster,
            centers=None,
            n_features=self.n_features,
            random_state=42,
        )
        self.df = pd.DataFrame({"id": np.arange(len(self.X)), "embeddings": self.X.tolist()})

    def _create_input_parquet_files(self, input_dir: str, npartitions: int = 2) -> None:
        """Create parquet files from the synthetic data for pipeline input."""
        os.makedirs(input_dir, exist_ok=True)
        chunk_size = len(self.df) // npartitions
        for i in range(npartitions):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < npartitions - 1 else len(self.df)
            chunk_df = self.df.iloc[start_idx:end_idx]
            chunk_path = os.path.join(input_dir, f"part_{i:04d}.parquet")
            chunk_df.to_parquet(chunk_path, index=False)

    def test_scale_dependent_reduces_duplicates(self, tmpdir: Path) -> None:
        """Verify tighter thresholds reduce semantic near-duplicate count at simulated large scale."""
        input_dir = os.path.join(tmpdir, "input")
        output_dir_base = os.path.join(tmpdir, "output_base")
        output_dir_scaled = os.path.join(tmpdir, "output_scaled")

        self._create_input_parquet_files(input_dir)
        executor = XennaExecutor()
        base_eps = 0.05

        # 1. Run Baseline (no scaling)
        pipeline_base = SemanticDeduplicationWorkflow(
            input_path=input_dir,
            output_path=output_dir_base,
            n_clusters=self.n_clusters,
            id_field="id",
            embedding_field="embeddings",
            distance_metric="cosine",
            which_to_keep="hard",
            eps=base_eps,
            random_state=42,
            verbose=False,
            scale_aware=False
        )
        results_base = pipeline_base.run(pairwise_executor=executor)
        duplicates_base = results_base.get_metadata("num_duplicates") or 0

        # 2. Run Scaled (simulated large scale)
        pipeline_scaled = SemanticDeduplicationWorkflow(
            input_path=input_dir,
            output_path=output_dir_scaled,
            n_clusters=self.n_clusters,
            id_field="id",
            embedding_field="embeddings",
            distance_metric="cosine",
            which_to_keep="hard",
            eps=base_eps,
            random_state=42,
            verbose=False,
            scale_aware=True,
            corpus_size_tokens=1_000_000_000_000, # 1T tokens
            target_model_params=100_000_000_000    # 100B params
        )
        results_scaled = pipeline_scaled.run(pairwise_executor=executor)
        duplicates_scaled = results_scaled.get_metadata("num_duplicates") or 0

        # The scaled pipeline should have a tighter threshold (smaller eps)
        # resulting in fewer duplicates identified.
        assert pipeline_scaled.eps < base_eps, "Epsilon was not tightened by scale_aware logic"
        assert duplicates_scaled < duplicates_base, (
            f"Expected fewer duplicates due to scale adjustment. "
            f"Base: {duplicates_base}, Scaled: {duplicates_scaled}"
        )
