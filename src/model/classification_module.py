from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.autograd import Variable


class DeepWeedsClassificationModule(pl.LightningModule):
    """
    PyTorch Lightning module for the DeepWeeds classification task.
    This module implements a convolutional neural network with a variable number of convolutional and fully connected layers.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        # Saves all arguments passed to the constructor as hyperparameters (self.hparams)
        self.save_hyperparameters()

        self.loss = torch.nn.CrossEntropyLoss()
        self.validation_step_outputs = []

        # Code from the project template for building the model

        in_channels = self.hparams.input_shape[0]

        # Compute the dimensions of the convolutional layers
        conv_channels: list[tuple[int, int]] = [
            (in_channels, self.hparams.n_channels_conv_0)
        ]
        for layer_i in range(1, self.hparams.n_conv_layers):
            previous_layer_i = layer_i - 1

            previous_dimensions = conv_channels[previous_layer_i]
            _, previous_out = previous_dimensions

            layer_in = previous_out
            layer_out = self.hparams.get(f"n_channels_conv_{layer_i}", previous_out * 2)

            conv_channels.append((layer_in, layer_out))

        # Built the layers
        layers = []
        for in_channels, out_channels in conv_channels:
            conv = self.conv_block(
                in_channels,
                out_channels,
                kernel_size=self.hparams.kernel_size,
                padding=(self.hparams.kernel_size - 1) // 2,
                use_bn=self.hparams.use_BN,
            )
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            layers.extend([conv, pool])

        self.conv_layers = nn.Sequential(*layers)
        self.pooling = (
            nn.AdaptiveAvgPool2d(1)
            if self.hparams.global_avg_pooling
            else nn.Identity()
        )
        self.output_size = self.hparams.num_classes

        fc_0_in = self.get_output_shape(
            self.conv_layers,
            self.pooling,
            shape=self.hparams.input_shape,
        )
        fc_0_out = self.hparams.n_channels_fc_0
        n_fc_layerss = self.hparams.n_fc_layers

        # Compute the dimenions
        fc_sizes = [(fc_0_in, fc_0_out)]
        for layer_i in range(1, n_fc_layerss):
            previous_layer_i = layer_i - 1

            _, fc_previous_out = fc_sizes[previous_layer_i]
            fc_in = fc_previous_out
            fc_out = self.hparams.get(f"n_channels_fc_{layer_i}", fc_previous_out // 2)

            fc_sizes.append((fc_in, fc_out))

        self.fc_layers = nn.ModuleList(
            [nn.Linear(int(n_in), int(n_out)) for n_in, n_out in fc_sizes]
        )

        _, last_fc_out = fc_sizes[-1]
        self.last_fc = nn.Linear(int(last_fc_out), self.output_size)
        self.dropout = nn.Dropout(p=self.hparams.dropout_rate)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Same as :meth:`torch.nn.Module.forward`.

        Code from the project template.

        Args:
            *args: Whatever you decide to pass into the forward method.
            **kwargs: Keyword arguments are also possible.

        Return:
            Your model's output
        """
        x = self.conv_layers(args[0])
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        for fc_layer in self.fc_layers:
            x = self.dropout(F.relu(fc_layer(x)))
        x = self.last_fc(x)
        return x

    def training_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        """Here you compute and return the training loss and some additional metrics for e.g. the progress bar or
        logger.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple dataloaders used)

        Return:
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
            - ``None`` - Skip to the next batch. This is only supported for automatic optimization.
                This is not supported for multi-GPU, TPU, IPU, or DeepSpeed.

        In this step you'd normally do the forward pass and calculate the loss for a batch.
        You can also do fancier things like multiple forward passes or something model specific.
        """
        outputs = self.forward(batch[0])
        loss = self.loss(outputs, batch[1])
        acc = self.accuracy(outputs, batch[1])[0]
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_acc", acc, sync_dist=True)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        """Operates on a single batch of data from the validation set. In this step you'd might generate examples
        or calculate anything of interest like accuracy.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple dataloaders used)

        Return:
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
            - ``None`` - Skip to the next batch.
        """
        outputs = self.forward(batch[0])
        loss = self.loss(outputs, batch[1])
        self.log("val_loss", loss, sync_dist=True)
        acc = self.accuracy(outputs, batch[1])[0]
        self.log("val_accuracy", acc, sync_dist=True)
        # Save the outputs to compute the mean at the end of the epoch
        self.validation_step_outputs.append(
            {
                "val_loss": loss,
                "val_accuracy": acc,
            }
        )
        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self):
        """Called when the val epoch ends.
        Calculates the mean loss and accuracy over all validation batches.

        Returns:
            _type_: _description_
        """
        avg_loss = torch.Tensor(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        avg_accuracy = torch.Tensor(
            [x["val_accuracy"] for x in self.validation_step_outputs]
        ).mean()
        self.log("val_loss_mean", avg_loss, sync_dist=True)
        self.log("val_accuracy_mean", avg_accuracy, sync_dist=True)
        self.validation_step_outputs.clear()
        return {"val_loss_mean": avg_loss, "val_accuracy_mean": avg_accuracy}

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        """Operates on a single batch of data from the test set. In this step you'd normally generate examples or
        calculate anything of interest such as accuracy.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple dataloaders used)

        Return:
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
            - ``None`` - Skip to the next batch.
        """
        outputs = self.forward(batch[0])
        loss = self.loss(outputs, batch[1])
        acc = self.accuracy(outputs, batch[1])[0]
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", acc, sync_dist=True)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self) -> Any:
        """Choose what optimizers and learning-rate schedulers to use in your optimization. Normally you'd need
        one. But in the case of GANs or similar you might have multiple. Optimization with multiple optimizers only
        works in the manual optimization mode.

        Return:
            Any of these 6 options.

            - **Single optimizer**.
            - **List or Tuple** of optimizers.
            - **Two lists** - The first list has multiple optimizers, and the second has multiple LR schedulers
              (or multiple ``lr_scheduler_config``).
            - **Dictionary**, with an ``"optimizer"`` key, and (optionally) a ``"lr_scheduler"``
              key whose value is a single LR scheduler or ``lr_scheduler_config``.
            - **None** - Fit will run without any optimizer.

        The ``lr_scheduler_config`` is a dictionary which contains the scheduler and its associated configuration.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate_init)

    def conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
    ) -> nn.Sequential:
        """Simple convolutional block.

        Code from the project template.

        :param in_channels:
            number of input channels
        :param out_channels:
            number of output channels
        :param kernel_size:
            kernel size
        :param stride:
            Stride of the convolution
        :param padding:
            padded value
        :param use_bn:
            if BN is applied

        :return: a convolutional block layer
        """
        conv_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        activation = nn.ReLU(inplace=False)
        if use_bn:
            batch_norm = nn.BatchNorm2d(out_channels)
            return nn.Sequential(conv_layer, activation, batch_norm)
        else:
            return nn.Sequential(conv_layer, activation)

    def get_output_shape(
        self,
        *layers: torch.nn.Sequential | torch.nn.Module,
        shape: tuple[int, int, int],
    ) -> int:
        """Calculate the output dimensions of a stack of conv layer.

        Code from the project template.

        Args:
            shape (tuple[int, int, int]): _description_

        Returns:
            int: _description_
        """
        channels, width, height = shape
        # pylint: disable=E1101
        var = Variable(torch.rand(1, channels, width, height))
        # pylint: enable=E1101

        seq = torch.nn.Sequential()
        for layer in layers:
            seq.append(layer)

        output_feat = seq(var)

        # Flatten the data out, and get it's size, this will be
        # the size of what's given to a fully connected layer
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def accuracy(self, output: torch.Tensor, target: torch.Tensor, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k.

        Code from the project template.

        Args:
            output (torch.Tensor): _description_
            target (torch.Tensor): _description_
            topk (tuple, optional): _description_. Defaults to (1,).

        Returns:
            _type_: _description_
        """
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(1.0 / batch_size))
        return res
