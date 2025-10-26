import torch
import torch.nn as nn
from typing import Optional
from pretrainedmodels import xception


class Xception(nn.Module):
    """
    Xception-based classification model.

    This model uses a pretrained Xception backbone, followed by global average pooling
    and two fully connected layers.

    Notes
    -----
    - Forward behavior:
        * If num_classes == 1 (binary), `forward` returns sigmoid probabilities in [0, 1].
        * If num_classes > 1 (multiclass), `forward` returns **raw logits** by default
          (suited for `nn.CrossEntropyLoss`).
        * You can override with `output_activation="softmax"` to return probabilities
          also in the multiclass case.
    - Use `predict_proba(x)` to consistently obtain probabilities for both cases.
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained: str = "imagenet",
        output_activation: str = "auto",
    ) -> None:
        """
        Initialize the Xception model.

        Parameters
        ----------
        num_classes : int, optional
            Number of output classes. If 1 -> binary; if >1 -> multiclass. Default is 1.
        pretrained : str, optional
            Pretrained weights to load for the backbone. Default is "imagenet".
        output_activation : {"auto", "none", "softmax", "sigmoid"}, optional
            Controls activation applied in `forward`:
            - "auto"   : sigmoid if num_classes==1, otherwise none (logits).
            - "none"   : always return logits.
            - "softmax": return softmax probabilities for multiclass (>1 classes).
            - "sigmoid": force sigmoid (mainly for multilabel setups; not typical for multiclass).
            Default is "auto".

        Raises
        ------
        ValueError
            If `num_classes < 1` or `output_activation` is invalid.
        """
        super().__init__()

        if num_classes < 1:
            raise ValueError("`num_classes` must be >= 1.")

        valid_acts = {"auto", "none", "softmax", "sigmoid"}
        if output_activation not in valid_acts:
            raise ValueError(f"`output_activation` must be one of {valid_acts}.")

        self.num_classes = num_classes
        self.output_activation = output_activation

        # Backbone
        self.base = xception(pretrained=pretrained)
        self.base.fc = nn.Identity()  # Remove the original fully connected layer

        # Head
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2048, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

        # Activations
        self._sigmoid = nn.Sigmoid()
        self._softmax = nn.Softmax(dim=1)

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation according to `num_classes` and `output_activation`."""
        act = self.output_activation

        if act == "none":
            return x

        if act == "sigmoid":
            return self._sigmoid(x)

        if act == "softmax":
            return self._softmax(x)

        # act == "auto"
        if self.num_classes == 1:
            return self._sigmoid(x)  # binary -> probabilities
        else:
            return x  # multiclass -> logits by default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Xception model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, H, W).

        Returns
        -------
        torch.Tensor
            - If `num_classes == 1` (binary, default "auto"): shape (batch_size, 1) with sigmoid probs.
            - If `num_classes > 1` and "auto"/"none": shape (batch_size, num_classes) logits.
            - If `num_classes > 1` and "softmax": probabilities over classes.
            - If "sigmoid" with num_classes>1, returns per-class independent probs (multilabel-like).
        """
        x = self.base.features(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # (B, 2048)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # logits for multiclass; pre-sigmoid for binary

        x = self._apply_activation(x)
        return x

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return probabilities for inference regardless of training-time activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, H, W).

        Returns
        -------
        torch.Tensor
            Probabilities:
            - Binary (num_classes==1): shape (batch_size, 1), sigmoid probabilities.
            - Multiclass (num_classes>1): shape (batch_size, num_classes), softmax probabilities.
        """
        self.eval()
        feats = self.base.features(x)
        feats = self.global_avg_pooling(feats)
        feats = feats.view(feats.size(0), -1)
        feats = self.dropout(feats)
        feats = self.fc1(feats)
        feats = self.relu(feats)
        logits = self.fc2(feats)

        if self.num_classes == 1:
            return self._sigmoid(logits)
        else:
            return self._softmax(logits)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate feature representation before the output layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, H, W).

        Returns
        -------
        torch.Tensor
            Feature embeddings of shape (batch_size, 128) after the first FC layer.

        Notes
        -----
        - Useful for downstream tasks such as clustering, visualization (t-SNE/UMAP),
          or training a separate classifier.
        """
        x = self.base.features(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        return x
