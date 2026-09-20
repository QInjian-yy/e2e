"""Independent Mean-ResNet downstream model."""

from torch import nn

from downstream.shared_model import SharedE2EModel


class MeanResNet(SharedE2EModel):
    model_name = "mean_resnet"
    pooling_name = "mean"
    downstream_description = "WSI downstream: [N,512] -> mean -> Linear(512,2)"

    def __init__(self, sr, model_spec=None):
        super().__init__(sr, model_spec)
        self.classifier = nn.Linear(512, 2)

    def aggregate_embeddings(self, embeddings):
        return embeddings.mean(dim=0, keepdim=True)

    def classify(self, wsi_embedding):
        return self.classifier(wsi_embedding)

    def downstream_gradient_groups(self):
        return {"WSI classifier": self.classifier.parameters()}

    # Compatibility for existing callers while the engine uses the new interface.
    def aggregate(self, embeddings):
        return self.aggregate_embeddings(embeddings)

