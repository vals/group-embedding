from typing import Iterable

import numpy as np
import torch
from torch.distributions.bernoulli import Bernoulli
from scvi import _CONSTANTS
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import (
    Encoder,
    FCLayers
)
from scvi.distributions import ZeroInflatedNegativeBinomial
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

torch.backends.cudnn.benchmark = True


class BinaryDecoder(torch.nn.Module):
    """ 
    Decodes data from latent space to logit space
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            **kwargs,
        )
    
        self.logit_decoder = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        """
        # Parameters for latent distribution
        p = self.decoder(x, *cat_list)
        p_logit = self.logit_decoder(p)
        return p_logit


class MyModule(BaseModuleClass):
    """
    Skeleton Variational auto-encoder model.

    Here we implement a basic version of scVI's underlying VAE [Lopez18]_.
    This implementation is for instructional purposes only.

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_batch = n_batch
        # this is needed to comply with some requirement of the VAEMixin class
        self.latent_distribution = "normal"

        # setup the parameters of your generative model, as well as your inference model

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = BinaryDecoder(
            n_latent,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

    def _get_inference_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        x = tensors[_CONSTANTS.X_KEY]

        input_dict = dict(x=x)
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]

        input_dict = {
            "z": z
        }
        return input_dict

    @auto_move_data
    def inference(self, x):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        # log the input to the variational distribution for numerical stability
        x_ = torch.log(1 + x)
        # get variational parameters via the encoder networks
        qz_m, qz_v, z = self.z_encoder(x_)

        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v)
        return outputs

    @auto_move_data
    def generative(self, z):
        """Runs the generative model."""

        # form the parameters of the Bernoulli likelihood
        px_logit = self.decoder(z)

        return dict(
            px_logit=px_logit
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
    ):
        x = tensors[_CONSTANTS.X_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        px_logit = generative_outputs["px_logit"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        reconst_loss = (
            -Bernoulli(logits=px_logit)
            .log_prob(x)
            .sum(dim=-1)
        )

        loss = torch.mean(reconst_loss + kl_divergence_z)

        kl_local = dict(
            kl_divergence_z=kl_divergence_z
        )
        kl_global = 0.0
        return LossRecorder(loss, reconst_loss, kl_local, kl_global)

    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
    ) -> np.ndarray:
        r"""
        Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.

        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale scamples to

        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        inference_kwargs = dict(n_samples=n_samples)
        inference_outputs, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]
        px_dropout = generative_outputs["px_dropout"]

        dist = ZeroInflatedNegativeBinomial(
            mu=px_rate, theta=px_r, zi_logits=px_dropout
        )

        if n_samples > 1:
            exprs = dist.sample().permute(
                [1, 2, 0]
            )  # Shape : (n_cells_batch, n_genes, n_samples)
        else:
            exprs = dist.sample()

        return exprs.cpu()

    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self, tensors, n_mc_samples):
        sample_batch = tensors[_CONSTANTS.X_KEY]
        local_l_mean = tensors[_CONSTANTS.LOCAL_L_MEAN_KEY]
        local_l_var = tensors[_CONSTANTS.LOCAL_L_VAR_KEY]

        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
            inference_outputs, generative_outputs, losses = self.forward(tensors)
            qz_m = inference_outputs["qz_m"]
            qz_v = inference_outputs["qz_v"]
            z = inference_outputs["z"]
            ql_m = inference_outputs["ql_m"]
            ql_v = inference_outputs["ql_v"]
            library = inference_outputs["library"]

            # Reconstruction Loss
            reconst_loss = losses.reconstruction_loss

            # Log-probabilities
            p_l = Normal(local_l_mean, local_l_var.sqrt()).log_prob(library).sum(dim=-1)
            p_z = (
                Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
                .log_prob(z)
                .sum(dim=-1)
            )
            p_x_zl = -reconst_loss
            q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
            q_l_x = Normal(ql_m, ql_v.sqrt()).log_prob(library).sum(dim=-1)

            to_sum[:, i] = p_z + p_l + p_x_zl - q_z_x - q_l_x

        batch_log_lkl = torch.logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        log_lkl = torch.sum(batch_log_lkl).item()
        return log_lkl
