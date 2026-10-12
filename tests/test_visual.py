import unittest

import numpy as np
from tensorgrad import Tensor

from tinydreamer.visual import VisualAutoencoder, reconstruction_metrics, weighted_binary_cross_entropy


class VisualTests(unittest.TestCase):
    def test_autoencoder_shapes_and_gradients(self):
        np.random.seed(0)
        model = VisualAutoencoder(hidden_size=12, latent_size=4)
        sample = Tensor(np.zeros((2, 10, 10, 6)))
        self.assertEqual(model.encode(sample).data.shape, (2, 4))
        logits = model(sample)
        self.assertEqual(logits.data.shape, (2, 10, 10, 6))
        weighted_binary_cross_entropy(logits, sample).backward()
        self.assertTrue(any(np.any(p.grad != 0) for p in model.parameters()))

    def test_metrics_detect_empty_prediction(self):
        targets = np.zeros((1, 10, 10, 6))
        targets[0, 0, 0, 0] = 1
        metrics = reconstruction_metrics(np.full_like(targets, -10), targets)
        self.assertEqual(metrics["foreground_recall"], 0.0)
        self.assertLess(metrics["pixel_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
