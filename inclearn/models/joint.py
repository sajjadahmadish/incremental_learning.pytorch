import collections
import logging

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from inclearn.lib import loops, factory, utils, network
from inclearn.lib.network import hook
from inclearn.models import IncrementalLearner
import torch.optim as optim
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Joint(IncrementalLearner):

    def __init__(self, args):
        super().__init__()

        self._disable_progressbar = args.get("no_progressbar", False)
        self._device = args["device"][0]
        self._multiple_devices = args["device"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._n_epochs = args["epochs"]
        self._scheduling = None
        self._opt_name = None
        self._n_classes = 0
        self._last_results = None

        logger.info("Initializing Joint")

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config", {
                "type": "pfc"
            }),
            device=self._device
        )

    def _before_task(self, data_loader, val_loader):
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), 'adam', self._lr
        )
        self._scheduler = optim.lr_scheduler.StepLR(self._optimizer, step_size=20, gamma=0.5)

    def _train_task(self, train_loader, val_loader):
        _, train_loader = self.inc_dataset.get_custom_loader(
            list(range(self._n_classes)), mode="train", data_source='train')

        logger.debug("nb {}.".format(len(train_loader.dataset)))
        logger.debug("_n_classes {}.".format(self._n_classes))

        self._training_step(train_loader, val_loader, 0, self._n_epochs)

    def _training_step(self, train_loader, val_loader, initial_epoch, nb_epochs):

        grad, act = None, None
        if len(self._multiple_devices) > 1:
            logger.info("Duplicating model on {} gpus.".format(len(self._multiple_devices)))
            training_network = nn.DataParallel(self._network, self._multiple_devices)
            if self._network.gradcam_hook:
                grad, act, back_hook, for_hook = hook.get_gradcam_hook(training_network)
                training_network.module.convnet.last_conv.register_backward_hook(back_hook)
                training_network.module.convnet.last_conv.register_forward_hook(for_hook)
        else:
            training_network = self._network

        for epoch in range(initial_epoch, nb_epochs):
            self._metrics = collections.defaultdict(float)

            self._epoch_percent = epoch / (nb_epochs - initial_epoch)

            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            for i, input_dict in enumerate(prog_bar, start=1):
                inputs, targets = input_dict["inputs"], input_dict["targets"]
                if len(inputs) == 1:
                    continue

                if grad is not None:
                    _clean_list(grad)
                    _clean_list(act)

                self._optimizer.zero_grad()
                loss = self._forward_loss(
                    training_network,
                    inputs,
                    targets,
                    gradcam_grad=grad,
                    gradcam_act=act
                )
                loss.backward()
                self._optimizer.step()

                self._print_metrics(prog_bar, epoch, nb_epochs, i)

            if self._scheduler:
                self._scheduler.step()

        if len(self._multiple_devices) == 1 and hasattr(training_network.convnet, "record_mode"):
            training_network.convnet.normal_mode()

    def _forward_loss(
            self,
            training_network,
            inputs,
            targets,
            gradcam_grad=None,
            gradcam_act=None,
            **kwargs
    ):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        # logger.debug(f'inputs: {inputs}')
        # logger.debug(f'targets: {targets}')

        outputs = training_network(inputs)
        if gradcam_act is not None:
            outputs["gradcam_gradients"] = gradcam_grad
            outputs["gradcam_activations"] = gradcam_act

        logits = outputs["logits"]
        t = torch.tensor(targets, dtype=torch.int64)
        loss = F.nll_loss(logits, t)

        # if not utils.check_loss(loss):
        #     raise ValueError("A loss is NaN: {}".format(self._metrics))
        self._metrics["loss"] += loss.item()

        return loss

    def _accuracy(self, data_loader):
        ypred, ytrue = self._eval_task(data_loader)
        ypred = ypred.argmax(dim=1)

        return 100 * round(np.mean(ypred == ytrue), 3)

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

    def _print_metrics(self, prog_bar, epoch, nb_epochs, nb_batches):
        pretty_metrics = ", ".join(
            "{}: {}".format(metric_name, round(metric_value / nb_batches, 3))
            for metric_name, metric_value in self._metrics.items()
        )

        prog_bar.set_description(
            "T{}/{}, E{}/{} => {}".format(
                self._task + 1, self._n_tasks, epoch + 1, nb_epochs, pretty_metrics
            )
        )


def _clean_list(l):
    for i in range(len(l)):
        l[i] = None
