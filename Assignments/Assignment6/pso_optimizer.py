import cupy as cp
import numpy as np

class BatchedGPUParticleSwarm:
    def __init__(
        self,
        n_particles: int,
        n_turbines: int,
        bounds: list[tuple[float, float]],
        maxiter: int = 100,
        w: float = 0.5,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: int = 42,
        disp: bool = False
    ):
        self.n_particles = n_particles
        self.n_turbines = n_turbines
        self.bounds = np.array(bounds)
        self.maxiter = maxiter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = cp.random.default_rng(seed)
        self.disp = disp
        
        self.best_global_pos = None
        self.best_global_score = cp.inf
        self.best_global_aep = 0.0
        
        self.lower_bound = cp.array(self.bounds[:, 0]).reshape(1, n_turbines, 2)
        self.upper_bound = cp.array(self.bounds[:, 1]).reshape(1, n_turbines, 2)

    def optimize(self, batched_objective_fn, callback=None):
        # Initialize positions
        pos = self.rng.uniform(
            low=self.lower_bound,
            high=self.upper_bound,
            size=(self.n_particles, self.n_turbines, 2)
        )
        vel = cp.zeros_like(pos)
        
        best_personal_pos = pos.copy()
        best_personal_score = cp.full(self.n_particles, cp.inf)
        
        for it in range(self.maxiter):
            # Evaluate objective function for the entire swarm
            scores, aeps = batched_objective_fn(pos)
            
            # Update personal bests
            improved_mask = scores < best_personal_score
            best_personal_score[improved_mask] = scores[improved_mask]
            best_personal_pos[improved_mask] = pos[improved_mask]
            
            # Update global best
            min_score_idx = int(cp.argmin(scores))
            if scores[min_score_idx] < self.best_global_score:
                self.best_global_score = float(scores[min_score_idx])
                self.best_global_pos = pos[min_score_idx].copy()
                self.best_global_aep = float(aeps[min_score_idx])
                
            if callback is not None:
                # Pass the CPU version of the best position
                flat_pos = self.best_global_pos.flatten()
                cpu_pos = flat_pos.get() if hasattr(flat_pos, 'get') else flat_pos
                callback(cpu_pos, self.best_global_score, self.best_global_aep, it)
                
            # Update velocities and positions
            r1 = self.rng.uniform(0, 1, size=(self.n_particles, self.n_turbines, 2))
            r2 = self.rng.uniform(0, 1, size=(self.n_particles, self.n_turbines, 2))
            
            vel = (
                self.w * vel + 
                self.c1 * r1 * (best_personal_pos - pos) + 
                self.c2 * r2 * (self.best_global_pos - pos)
            )
            pos = pos + vel
            
            # Apply bounds reflection
            out_lower = pos < self.lower_bound
            out_upper = pos > self.upper_bound
            
            pos = cp.where(out_lower, 2 * self.lower_bound - pos, pos)
            pos = cp.where(out_upper, 2 * self.upper_bound - pos, pos)
            pos = cp.clip(pos, self.lower_bound, self.upper_bound)
            
            vel = cp.where(out_lower | out_upper, -0.5 * vel, vel)
            
            if self.disp:
                print(f"For {self.n_turbines} turbines:  PSO Iter {it+1:02d}/{self.maxiter} | Best Score: {self.best_global_score:>10.1f} | True AEP: {self.best_global_aep:.3f} GWh")
                
        flat_best_pos = self.best_global_pos.flatten()
        return flat_best_pos.get() if hasattr(flat_best_pos, 'get') else flat_best_pos, self.best_global_score
