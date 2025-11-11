from mani_skill.utils import common
import numpy as np


def embed_one(agent, dataset, sample_id):
    obs = dataset.__getitem__(sample_id)["observations"]
    bb = common.to_tensor(obs)
    bb['state'] = bb['state'].unsqueeze(0)
    bb['point_forces'] = bb['point_forces'].unsqueeze(0)
    emb = agent.encode_obs(bb, eval_mode=True)
    femb1 = emb[0, :32]
    femb2 = emb[0, 80:112]
    return femb1, femb2


def embed_trajectory(agent, dataset, trajectory_id):
    embedded_trajectory = []
    first_col = np.array(dataset.slices)[:, 0]
    change_indices = np.insert(np.where(np.diff(first_col) != 0)[0] + 1, 0, 0)
    s, e = change_indices[trajectory_id:trajectory_id + 2]
    for i in range(s, e):
        femb1, femb2 = embed_one(agent, dataset, i)
        embedded_trajectory.append(femb1.cpu().detach().numpy())
    
    return embedded_trajectory
