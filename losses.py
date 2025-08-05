# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import flax
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from utils import batch_mul

import time
import tracemalloc


def get_optimizer(config):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = flax.optim.Adam(beta1=config.optim.beta1, eps=config.optim.eps,
                                weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(state,
                  grad,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    lr = state.lr
    if warmup > 0:
      lr = lr * jnp.minimum(state.step / warmup, 1.0)
    if grad_clip >= 0:
      # Compute global gradient norm
      grad_norm = jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(grad)]))
      # Clip gradient
      clipped_grad = jax.tree_map(
        lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad)
    else:  # disabling gradient clipping if grad_clip < 0
      clipped_grad = grad
    return state.optimizer.apply_gradient(clipped_grad, learning_rate=lr)

  return optimize_fn


def reg_fn(score_fn, x, labels, rng, k: int, strategy: str, vector: str, moment: str):

    if k <= 0:
        # shape (batch,)
        return jnp.zeros(x.shape[0], dtype=x.dtype), 0, 0

    def sample_noise(key):
        if vector == 'rademacher':
            return jax.random.rademacher(key, x.shape).astype(x.dtype)
        elif vector == 'gaussian':
            return jax.random.normal(key, x.shape).astype(x.dtype)
        else:
            raise ValueError(f"Unknown vector type {vector!r}")

    def compute_trace(eps, jvp):
        trace = jnp.sum(eps * jvp, axis=tuple(range(1, x.ndim)))
        if moment == 'first':
            return trace
        elif moment == 'second':
            return trace**2
        else:
            raise ValueError(f"Unknown moment {moment!r}")

    tracemalloc.start()
    t0 = time.time()

    if strategy == 'memory':
        acc = jnp.zeros(x.shape[0], dtype=x.dtype)
        for _ in range(k):
            rng, key = jax.random.split(rng)
            eps = sample_noise(key)
            _, jvp = jax.jvp(lambda v: score_fn(v, labels), (x,), (eps,))
            acc = acc + compute_trace(eps, jvp)
        result = acc / k

    elif strategy == 'vmap':
        rngs = jax.random.split(rng, k)
        def single(key):
            eps = sample_noise(key)
            _, jvp = jax.jvp(lambda v: score_fn(v, labels), (x,), (eps,))
            return compute_trace(eps, jvp)
        traces = jax.vmap(single)(rngs)  # shape (k, batch)
        result = jnp.mean(traces, axis=0)

    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    # stop measuring
    elapsed = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, elapsed, peak


def get_sde_loss_fn(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5,
            k = 0, gamma = 0.0, strategy = 'memory', vector = 'rademacher', moment = 'second',
            comp_k = 0, comp_strategy = 'memory', comp_vector = 'rademacher', comp_moment = 'second'):

  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, states, batch):
    """Compute the loss function.

    Args:
      rng: A JAX random state.
      params: A dictionary that contains trainable parameters of the score-based model.
      states: A dictionary that contains mutable states of the score-based model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
      new_model_state: A dictionary that contains the mutated states of the score-based model.
    """

    score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous, return_state=True)
    data = batch['image']

    rng, step_rng = random.split(rng)
    t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)
    rng, step_rng = random.split(rng)
    z = random.normal(step_rng, data.shape)
    mean, std = sde.marginal_prob(data, t)
    perturbed_data = mean + batch_mul(std, z)
    rng, step_rng = random.split(rng)
    score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)

    if not likelihood_weighting:
      losses = jnp.square(batch_mul(score, std) + z)
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
    else:
      g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
      losses = jnp.square(score + batch_mul(z, 1. / std))
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

    base_loss = jnp.mean(losses)

    def pure_score(x_in, t_in):
      return score_fn(x_in, t_in, rng=step_rng)[0]

    rng, reg_rng = random.split(rng)
    raw_reg, elapsed, peak = reg_fn(
      score_fn=pure_score,
      x=perturbed_data,
      labels=t,
      rng=reg_rng,
      k=k,
      strategy=strategy,
      moment=moment,
      vector=vector
    )
    reg = jnp.mean(raw_reg)

    loss = base_loss + gamma * reg

    comp_raw_reg, comp_elapsed, comp_peak = reg_fn(
      score_fn=pure_score,
      x=perturbed_data,
      labels=t,
      rng=reg_rng,
      k=comp_k,
      strategy=comp_strategy,
      moment=comp_moment,
      vector=comp_vector
    )
    comp_reg = jnp.mean(comp_raw_reg)

    return loss, (new_model_state, base_loss, reg, elapsed, peak, comp_reg, comp_elapsed, comp_peak)

  return loss_fn


def get_smld_loss_fn(vesde, model, train, reduce_mean=False,
                     k=0, gamma=0.0, strategy='memory', vector='rademacher', moment='second',
                     comp_k=0, comp_strategy='memory', comp_vector='rademacher', comp_moment='second'):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = vesde.discrete_sigmas[::-1]
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, states, batch):
    model_fn = mutils.get_model_fn(model, params, states, train=train)
    data = batch['image']
    rng, step_rng = random.split(rng)
    labels = random.choice(step_rng, vesde.N, shape=(data.shape[0],))
    sigmas = smld_sigma_array[labels]
    rng, step_rng = random.split(rng)
    noise = batch_mul(random.normal(step_rng, data.shape), sigmas)
    perturbed_data = noise + data
    rng, step_rng = random.split(rng)
    score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
    target = -batch_mul(noise, 1. / (sigmas ** 2))
    losses = jnp.square(score - target)
    losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * sigmas ** 2

    base_loss = jnp.mean(losses)

    def pure_score(x_in, lbl_in):
      return model_fn(x_in, lbl_in, rng=step_rng)[0]

    rng, reg_rng = random.split(rng)
    raw_reg, elapsed, peak = reg_fn(
      score_fn=pure_score,
      x=perturbed_data,
      labels=labels,
      rng=reg_rng,
      k=k,
      strategy=strategy,
      moment=moment,
      vector=vector
    )
    reg = jnp.mean(raw_reg)

    loss = base_loss + gamma * reg

    comp_raw_reg, comp_elapsed, comp_peak = reg_fn(
        score_fn=pure_score,
        x=perturbed_data,
        labels=labels,
        rng=reg_rng,
        k=comp_k,
        strategy=comp_strategy,
        moment=comp_moment,
        vector=comp_vector
    )
    comp_reg = jnp.mean(comp_raw_reg)

    return loss, (new_model_state, base_loss, reg, elapsed, peak, comp_reg, comp_elapsed, comp_peak)

  return loss_fn


def get_ddpm_loss_fn(vpsde, model, train, reduce_mean=True,
                     k=0, gamma=0.0, strategy='memory', vector='rademacher', moment='second',
                     comp_k=0, comp_strategy='memory', comp_vector='rademacher', comp_moment='second'):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, states, batch):
    model_fn = mutils.get_model_fn(model, params, states, train=train)
    data = batch['image']
    rng, step_rng = random.split(rng)
    labels = random.choice(step_rng, vpsde.N, shape=(data.shape[0],))
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, data.shape)
    perturbed_data = batch_mul(sqrt_alphas_cumprod[labels], data) + \
                     batch_mul(sqrt_1m_alphas_cumprod[labels], noise)
    rng, step_rng = random.split(rng)
    score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
    losses = jnp.square(score - noise)
    losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)

    base_loss = jnp.mean(losses)

    def pure_score(x_in, lbl_in):
      return model_fn(x_in, lbl_in, rng=step_rng)[0]

    rng, reg_rng = random.split(rng)
    raw_reg, elapsed, peak = reg_fn(
      score_fn=pure_score,
      x=perturbed_data,
      labels=labels,
      rng=reg_rng,
      k=k,
      strategy=strategy,
      moment=moment,
      vector=vector
    )
    reg = jnp.mean(raw_reg)

    loss = base_loss + gamma * reg

    comp_raw_reg, comp_elapsed, comp_peak = reg_fn(
        score_fn=pure_score,
        x=perturbed_data,
        labels=labels,
        rng=reg_rng,
        k=comp_k,
        strategy=comp_strategy,
        moment=comp_moment,
        vector=comp_vector
    )
    comp_reg = jnp.mean(comp_raw_reg)

    return loss, (new_model_state, base_loss, reg, elapsed, peak, comp_reg, comp_elapsed, comp_peak)

  return loss_fn


def get_step_fn(sde, model, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False,
                k: int=0, gamma: float=0.0, strategy: str='memory', vector: str='rademacher', moment: str='second',
                comp_k: int=0, comp_strategy: str='memory', comp_vector: str='rademacher', comp_moment: str='first'):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training and `False` for evaluation.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, model, train, reduce_mean=reduce_mean, continuous=True, likelihood_weighting=likelihood_weighting,
                              k=k, gamma=gamma, strategy=strategy, vector=vector, moment=moment,
                              comp_k=comp_k, comp_strategy=comp_strategy, comp_vector=comp_vector, comp_moment=comp_moment)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, model, train, reduce_mean=reduce_mean,
                                 k=k, gamma=gamma, strategy=strategy, vector=vector, moment=moment,
                                 comp_k=comp_k, comp_strategy=comp_strategy, comp_vector=comp_vector,comp_moment=comp_moment)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, model, train, reduce_mean=reduce_mean,
                                 k=k, gamma=gamma, strategy=strategy, vector=vector, moment=moment,
                                 comp_k=comp_k, comp_strategy=comp_strategy, comp_vector=comp_vector,comp_moment=comp_moment)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(carry_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
      batch: A mini-batch of training/evaluation data.

    Returns:
      new_carry_state: The updated tuple of `carry_state`.
      loss: The average loss value of this state.
    """

    (rng, state) = carry_state
    rng, step_rng = jax.random.split(rng)
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    if train:
      params = state.optimizer.target
      states = state.model_state
      (loss, (new_model_state, base_loss, reg, elapsed, peak, comp_reg, comp_elapsed, comp_peak)), grad = grad_fn(step_rng, params, states, batch)
      grad = jax.lax.pmean(grad, axis_name='batch')
      new_optimizer = optimize_fn(state, grad)
      new_params_ema = jax.tree_multimap(
        lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
        state.params_ema, new_optimizer.target
      )
      step = state.step + 1
      new_state = state.replace(
        step=step,
        optimizer=new_optimizer,
        model_state=new_model_state,
        params_ema=new_params_ema
      )
    else:
      (loss, (_, base_loss, reg, elapsed, peak, comp_reg, comp_elapsed, comp_peak)) = loss_fn(step_rng, state.params_ema, state.model_state, batch)
      new_state = state

    loss = jax.lax.pmean(loss, axis_name='batch')
    base_loss = jax.lax.pmean(base_loss, axis_name='batch')
    reg = jax.lax.pmean(reg, axis_name='batch')
    elapsed = jax.lax.pmean(elapsed, axis_name='batch')
    peak = jax.lax.pmean(peak, axis_name='batch')
    comp_reg = jax.lax.pmean(comp_reg, axis_name='batch')
    comp_elapsed = jax.lax.pmean(comp_elapsed, axis_name='batch')
    comp_peak = jax.lax.pmean(comp_peak, axis_name='batch')

    new_carry_state = (rng, new_state)
    return new_carry_state, {'loss': loss, 'base': base_loss, 'reg': reg, 'elapsed': elapsed, 'peak': peak,
                             'comp reg': comp_reg, 'comp elapsed': comp_elapsed, 'comp peak': comp_peak}

  return step_fn
