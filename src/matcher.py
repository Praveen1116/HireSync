from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from src.skills import analyze_skill_gap

# Path to trained model weights
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "learned_scorer.pth"


class ResumeJDScorer(nn.Module):
    """
    Lightweight PyTorch MLP that takes interaction features
    between Resume and JD embeddings + skill overlap, outputting
    a calibrated match probability [0.0, 1.0].
    """
    def __init__(self, input_dim=1537):
        super(ResumeJDScorer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class HireSyncMatcher:
    """
    Inference engine that generates SBERT embeddings,
    computes baseline cosine similarity, and predicts the learned match score.
    """
    def __init__(self, model_path=DEFAULT_MODEL_PATH, sbert_name="all-MiniLM-L6-v2"):
        self.device = torch.device("cpu")  # CPU is fast and lightweight for inference

        # 1. Load Sentence-BERT embedding model
        print("Loading SBERT embedding model...")
        self.sbert = SentenceTransformer(sbert_name, device=self.device)

        # 2. Load trained PyTorch learned scorer weights
        print(f"Loading trained weights from: {model_path}")
        self.model = ResumeJDScorer(input_dim=1537).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def get_embedding(self, text: str) -> np.ndarray:
        """Generates normalized 384-dimensional dense vector for given text."""
        return self.sbert.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    def match(self, resume_text: str, jd_text: str) -> dict:
        """
        Runs full matching pipeline:
        - Extracts skills and computes skill gap
        - Computes baseline cosine similarity score (0 - 100%)
        - Predicts learned neural network match score (0 - 100%)
        """
        # 1. Analyze skills gap
        skill_analysis = analyze_skill_gap(resume_text, jd_text)
        overlap_ratio = skill_analysis["overlap_ratio"]

        # 2. Generate SBERT embeddings
        u = self.get_embedding(resume_text).reshape(1, -1)
        v = self.get_embedding(jd_text).reshape(1, -1)

        # 3. Compute Baseline Cosine Score (0.0 to 100.0)
        cosine_sim = float(np.sum(u * v))
        baseline_score = max(0.0, min(100.0, cosine_sim * 100.0))

        # 4. Construct Cross-Feature Vector: [u, v, |u - v|, u * v, overlap_ratio]
        diff = np.abs(u - v)
        prod = u * v
        overlap_feature = np.array([[overlap_ratio]])
        x_features = np.hstack([u, v, diff, prod, overlap_feature])

        # 5. Predict with Learned PyTorch Neural Network
        with torch.no_grad():
            x_tensor = torch.tensor(x_features, dtype=torch.float32).to(self.device)
            learned_prob = float(self.model(x_tensor).item())
            learned_score = max(0.0, min(100.0, learned_prob * 100.0))

        # 6. Qualitative Category
        if learned_score >= 70.0:
            category = "Strong Match"
        elif learned_score >= 45.0:
            category = "Potential Match"
        else:
            category = "Low Match"

        return {
            "learned_score": round(learned_score, 1),
            "baseline_cosine_score": round(baseline_score, 1),
            "category": category,
            "skill_gap": skill_analysis
        }

