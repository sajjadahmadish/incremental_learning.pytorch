import pdb

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.nn import functional as F
from tqdm import tqdm

from inclearn.lib import factory, herding, network, utils
from inclearn.models.base import IncrementalLearner

EPSILON = 1e-8


class ICarl(IncrementalLearner):
    """Implementation of iCarl.

    # References:
    - iCaRL: Incremental Classifier and Representation Learning
      Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
      https://arxiv.org/abs/1611.07725

    :param args: An argparse parsed arguments object.
    """

    def __init__(self, args):
        super().__init__()

        self._device = args["device"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._n_classes = 0

        self._network = network.BasicNet(
            args["convnet"],
            device=self._device,
            use_bias=True,
            extract_no_act=True,
            classifier_no_act=False
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._distil_loss = F.binary_cross_entropy_with_logits

        self._herding_indexes = []

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

    # ----------
    # Public API
    # ----------

    def _before_task(self, train_loader, val_loader):
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)
        print("Now {} examplars per class.".format(self._memory_per_class))

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )

        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, self._scheduling, gamma=self._lr_decay
        )

    def _train_task(self, train_loader, val_loader):
        print("nb ", len(train_loader.dataset))

        for epoch in range(self._n_epochs):
            _loss, val_loss = 0., 0.

            self._scheduler.step()

            prog_bar = tqdm(train_loader)
            for i, (inputs, targets, memory_flags) in enumerate(prog_bar, start=1):
                self._optimizer.zero_grad()

                loss = self._forward_loss(inputs, targets, memory_flags)
                loss.backward()

                self._optimizer.step()

                _loss += loss.item()

                if val_loader is not None and i == len(train_loader):
                    for inputs, targets, memory_flags in val_loader:
                        val_loss += self._forward_loss(inputs, targets, memory_flags).item()

                prog_bar.set_description(
                    "Task {}/{}, Epoch {}/{} => Clf loss: {}, Val loss: {}".format(
                        self._task + 1, self._n_tasks, epoch + 1, self._n_epochs,
                        round(_loss / i, 3), round(val_loss, 3)
                    )
                )

    def _forward_loss(self, inputs, targets, memory_flags):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        onehot_targets = utils.to_onehot(targets, self._n_classes).to(self._device)
        logits = self._network(inputs)

        loss = self._compute_loss(inputs, logits, targets, onehot_targets, memory_flags)

        if not utils._check_loss(loss):
            pdb.set_trace()

        return loss

    def _after_task(self, inc_dataset):
        self.build_examplars(inc_dataset)

        self._old_model = self._network.copy().freeze()

    def _eval_task(self, data_loader):
        ypred, ytrue = self.compute_accuracy(self._network, data_loader, self._class_means)

        return ypred, ytrue

    # -----------
    # Private API
    # -----------

    def _compute_loss(self, inputs, logits, targets, onehot_targets, memory_flags):
        if self._old_model is None:
            loss = F.binary_cross_entropy_with_logits(logits, onehot_targets)
        else:
            old_targets = torch.sigmoid(self._old_model(inputs).detach())

            new_targets = onehot_targets.clone()
            new_targets[..., :-self._task_size] = old_targets

            loss = F.binary_cross_entropy_with_logits(logits, new_targets)

        return loss

    def _compute_predictions(self, data_loader):
        preds = torch.zeros(self._n_train_data, self._n_classes, device=self._device)

        for idxes, inputs, _ in data_loader:
            inputs = inputs.to(self._device)
            idxes = idxes[1].to(self._device)

            preds[idxes] = self._network(inputs).detach()

        return torch.sigmoid(preds)

    def _classify(self, data_loader):
        if self._means is None:
            raise ValueError(
                "Cannot classify without built examplar means,"
                " Have you forgotten to call `before_task`?"
            )
        if self._means.shape[0] != self._n_classes:
            raise ValueError(
                "The number of examplar means ({}) is inconsistent".format(self._means.shape[0]) +
                " with the number of classes ({}).".format(self._n_classes)
            )

        ypred = []
        ytrue = []

        for _, inputs, targets in data_loader:
            inputs = inputs.to(self._device)

            features = self._network.extract(inputs).detach()
            preds = self._get_closest(self._means, F.normalize(features))

            ypred.extend(preds)
            ytrue.extend(targets)

        return np.array(ypred), np.array(ytrue)

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    # -----------------
    # Memory management
    # -----------------

    def build_examplars(self, inc_dataset):
        print("Building & updating memory.")

        data_memory, targets_memory = [], []
        self._class_means = np.zeros((100, self._network.features_dim))

        for class_idx in range(self._n_classes):
            # We extract the features, both normal and flipped:
            inputs, loader = inc_dataset.get_custom_loader(class_idx, mode="test")
            features, targets = utils.extract_features(self._network, loader)
            features_flipped, _ = utils.extract_features(
                self._network,
                inc_dataset.get_custom_loader(class_idx, mode="flip")[1]
            )

            if class_idx >= self._n_classes - self._task_size:
                # New class, selecting the examplars:
                memory = self.get_memory() if self._task > 0 else None
                self._herding_indexes.append(
                    #herding.minimize_confusion(
                    #    inc_dataset, self._network, memory, class_idx, self._memory_per_class
                    #)
                    herding.icarl_selection(features, self._memory_per_class)
                )

            # Reducing examplars:
            selected_indexes = self._herding_indexes[class_idx][:self._memory_per_class]
            self._herding_indexes[class_idx] = selected_indexes

            # Re-computing the examplar mean (which may have changed due to the training):
            examplar_mean = self.compute_examplar_mean(
                features, features_flipped, selected_indexes, self._memory_per_class
            )

            data_memory.append(inputs[selected_indexes])
            targets_memory.append(targets[selected_indexes])

            self._class_means[class_idx, :] = examplar_mean

        self._data_memory = np.concatenate(data_memory)
        self._targets_memory = np.concatenate(targets_memory)

    def get_memory(self):
        return self._data_memory, self._targets_memory

    @staticmethod
    def compute_examplar_mean(feat_norm, feat_flip, indexes, nb_max):
        D = feat_norm.T
        D = D / (np.linalg.norm(D, axis=0) + EPSILON)

        D2 = feat_flip.T
        D2 = D2 / (np.linalg.norm(D2, axis=0) + EPSILON)

        selected_d = D[..., indexes]
        selected_d2 = D2[..., indexes]

        mean = (np.mean(selected_d, axis=1) + np.mean(selected_d2, axis=1)) / 2
        mean /= (np.linalg.norm(mean) + EPSILON)

        return mean

    @staticmethod
    def compute_accuracy(model, loader, class_means):
        features, targets_ = utils.extract_features(model, loader)

        targets = np.zeros((targets_.shape[0], 100), np.float32)
        targets[range(len(targets_)), targets_.astype('int32')] = 1.

        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

        # Compute score for iCaRL
        sqd = cdist(class_means, features, 'sqeuclidean')
        score_icarl = (-sqd).T

        return np.argsort(score_icarl, axis=1)[:, -1], targets_
