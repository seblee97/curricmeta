from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import torch

from curricmeta.meta.outer_loop_base import OuterLoop
from curricmeta.meta.supervised_curriculum_inner import SupervisedCurriculumInnerLoop
from curricmeta.tasks.base import SupervisedStagedTask
from curricmeta.curriculum.base import Curriculum
from curricmeta.utils.registry import register

BuildTaskFn = Callable[[], SupervisedStagedTask]
BuildCurriculumFn = Callable[[], Curriculum]
BuildModelFn = Callable[[SupervisedStagedTask], torch.nn.Module]


@register("outer_loop", "es_supervised")
class ESSupervisedOuterLoop(OuterLoop):
    """
    ES outer loop for generic staged supervised tasks with curricula.

    It only knows:
      - how to build fresh (task, curriculum, model)
      - how many stages there are (from task.num_stages())
      - how to call the inner loop
    """

    def __init__(
        self,
        config: Dict[str, Any],
        inner_loop: SupervisedCurriculumInnerLoop,
        build_task: BuildTaskFn,
        build_curriculum: BuildCurriculumFn,
        build_model: BuildModelFn,
        device: torch.device,
    ):
        super().__init__(config)
        self.inner_loop = inner_loop
        self.build_task = build_task
        self.build_curriculum = build_curriculum
        self.build_model = build_model
        self.device = device

        # ES hyperparameters
        self.n_iters: int = int(config.get("n_iters", 20))
        self.population_size: int = int(config.get("population_size", 16))
        self.sigma: float = float(config.get("sigma", 0.1))
        self.step_size: float = float(config.get("step_size", 0.1))
        self.init_log_lr: float = float(config.get("init_log_lr", -3.0))

        # Infer number of stages from a sample task
        sample_task = self.build_task()
        sample_task.setup()
        n_stages = sample_task.num_stages()
        sample_task.teardown()

        self.n_stages: int = n_stages
        self.theta: torch.Tensor = torch.full(
            (self.n_stages,), self.init_log_lr, dtype=torch.float32
        )

    def initialize_meta_params(self) -> Dict[str, Any]:
        with torch.no_grad():
            per_stage_lr = torch.exp(self.theta).tolist()
        return {"per_stage_lr": per_stage_lr}

    def _sample_population(self) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        eps = torch.randn(self.population_size, self.n_stages)
        meta_param_list: List[Dict[str, Any]] = []

        with torch.no_grad():
            for j in range(self.population_size):
                theta_j = self.theta + self.sigma * eps[j]
                per_stage_lr = torch.exp(theta_j).tolist()
                meta_param_list.append({"per_stage_lr": per_stage_lr})

        return eps, meta_param_list

    def _evaluate_one(self, meta_params: Dict[str, Any]) -> float:
        task = self.build_task()
        curriculum = self.build_curriculum()
        model = self.build_model(task)

        inner_results = self.inner_loop.run(
            task=task,
            curriculum=curriculum,
            model=model,
            meta_params=meta_params,
            device=self.device,
        )

        return float(inner_results["test_acc"])

    def _evaluate_population(
        self,
        meta_param_list: List[Dict[str, Any]],
    ) -> List[float]:
        return [self._evaluate_one(mp) for mp in meta_param_list]

    def step(
        self,
        inner_results: Dict[str, Any],
        meta_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Not used in this ES variant; present to satisfy the interface.
        return meta_params

    def run(self) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        best_reward = float("-inf")
        best_meta_params: Dict[str, Any] = self.initialize_meta_params()

        for t in range(self.n_iters):
            eps, meta_param_list = self._sample_population()
            rewards = self._evaluate_population(meta_param_list)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            avg_reward = rewards_tensor.mean().item()
            max_reward = rewards_tensor.max().item()

            if max_reward > best_reward:
                best_reward = max_reward
                j_best = int(rewards_tensor.argmax().item())
                best_meta_params = meta_param_list[j_best]

            norm_rewards = (rewards_tensor - rewards_tensor.mean()) / (
                rewards_tensor.std() + 1e-8
            )
            grad_est = (
                1.0
                / (self.population_size * self.sigma)
                * torch.matmul(norm_rewards, eps)
            )

            with torch.no_grad():
                self.theta += self.step_size * grad_est

            history.append(
                {"iter": t, "avg_reward": avg_reward, "max_reward": max_reward}
            )

        return {
            "history": history,
            "best_meta_params": best_meta_params,
        }
