import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import sys
import os
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.nlhe_trainer import NLHESelfPlayTrainer, NLHETrainingConfig

def worker_fn(args):
    seed, state_dict = args
    config = NLHETrainingConfig(hands_per_epoch=4, device="cpu")
    trainer = NLHESelfPlayTrainer(config, seed=seed)
    
    # Rebuild PyTorch state dicts from NumPy
    sd_policy_t = {k: torch.from_numpy(v) for k, v in state_dict[0].items()}
    sd_opp_t = {k: torch.from_numpy(v) for k, v in state_dict[1].items()}
    
    trainer.policy.load_state_dict(sd_policy_t)
    trainer.opponent_encoder.load_state_dict(sd_opp_t)
    
    import io
    # Play exactly one hand
    experiences = trainer._play_hand(use_search=False)
    
    # Serialize to raw bytes to bypass PyTorch's broken multiprocessing shm manager
    buf = io.BytesIO()
    torch.save(experiences, buf)
    return buf.getvalue()

def test_mp():
    import numpy as np
    config = NLHETrainingConfig(hands_per_epoch=4, device="cpu")
    master = NLHESelfPlayTrainer(config, seed=42)
    
    # Export state dict to plain NumPy arrays to bypass PyTorch multiprocessing bugs on Mac
    sd_policy = {k: v.cpu().numpy() for k, v in master.policy.state_dict().items()}
    sd_opp = {k: v.cpu().numpy() for k, v in master.opponent_encoder.state_dict().items()}
    sd = (sd_policy, sd_opp)
    
    import io
    try:
        with ProcessPoolExecutor(max_workers=2) as pool:
            results_bytes = list(pool.map(worker_fn, [(i, sd) for i in range(2)]))
            
        results = [torch.load(io.BytesIO(b), weights_only=False) for b in results_bytes]
        total = sum(len(r) for r in results)
        print(f"Success! Collected {total} experiences.")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_mp()
