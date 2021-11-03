import logging

import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import factory, loops, network, utils
from inclearn.models import IncrementalLearner

logger = logging.getLogger(__name__)


class Joint(IncrementalLearner):

    def __init__(self, args):
        self._device = args["device"][0]
        self._multiple_devices = args["device"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._lr_decay = args["lr_decay"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]
        self._scheduling = None
        self._n_classes = 0

        self._data_memory, self._targets_memory = None, None

        logger.info("Initializing Joint")

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config", {
                "type": "fc",
                "use_bias": True
            }),
            device=self._device
        )

    def _before_task(self, data_loader, val_loader):
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=20, gamma=self._lr_decay)

    def _train_task(self, train_loader, val_loader):
        logger.debug("nb {}.".format(len(train_loader.dataset)))
        logger.debug("_n_classes {}.".format(self._n_classes))

        self._data_memory = train_loader.dataset.x[:]
        self._targets_memory = train_loader.dataset.y[:]

        epoch = self._n_epochs[0] if self._task == 0 else self._n_epochs[1]

        loops.single_loop(
            train_loader,
            val_loader,
            self._multiple_devices,
            self._network,
            epoch,
            self._optimizer,
            scheduler=self._scheduler,
            train_function=self._forward_loss,
            eval_function=self._accuracy,
            task=self._task,
            n_tasks=self._n_tasks,
            skip_last_batch_if_one=True
        )

    def _after_task(self, inc_dataset):
        self._network.on_task_end()

    def _eval_task(self, data_loader):
        logger.debug("nb-test {}.".format(len(data_loader.dataset)))

        ypred = []
        ytrue = []

        for data in data_loader:
            with torch.no_grad():
                out = self._network(data["inputs"].to(self._device))["logits"]

            ytrue.append(data["targets"].numpy())
            ypred.append(out.cpu().numpy())

        ytrue = np.concatenate(ytrue)
        ypred = np.concatenate(ypred)
        return ypred, ytrue

    def _accuracy(self, loader):
        ypred, ytrue = self._eval_task(loader)
        ypred = ypred.argmax(dim=1)

        return 100 * round(np.mean(ypred == ytrue), 3)

    def _forward_loss(self, training_network, inputs, targets, memory_flags, metrics, **kwargs):
        inputs, targets = inputs.to(self._device), targets.to(self._device)

        outputs = training_network(inputs)

        logits = outputs["logits"]

        loss = F.cross_entropy(logits, targets)
        metrics["clf"] += loss.item()

        # if not utils.check_loss(loss):
        #     raise ValueError("Loss became invalid ({}).".format(loss))

        metrics["loss"] += loss.item()

        return loss

    def get_memory(self):
        return self._data_memory, self._targets_memory
