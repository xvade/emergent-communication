"""JAX reimplementation of the `communication/toy` Melting Pot substrate.

Source of truth: `meltingpot/meltingpot/configs/substrates/communication/toy.py`
and `meltingpot/meltingpot/lua/levels/communication/toy/components.lua` in the
xvade/meltingpot fork, commit 390e2b9. This port follows those files. Where it
can't, it takes the behaviour from Melting Pot's engine code:

* Map: two walled 5x4 rooms. Apple room: apple at (3,1), spawn at (3,2).
  Pad room: upper pad (16,1), spawn (16,2), lower pad (16,4).
* Players: 2, sharing one spawn group, so the room is chosen at random each
  episode. Initial orientation is random (Avatar `randomizeInitialOrientation`
  defaults to true in avatar_library.lua).
* Apple: present with probability `exists_prob`. If present the upper pad is
  active, otherwise the lower pad. Both pads look identical in either state,
  and an absent apple is invisible.
* Stepping onto either pad ends the episode. If that pad is active, every
  player gets `reward` (1).
* Communication: one global bit. It is 1 for the observation produced by a
  step if any player chose COMMUNICATE in that step (BinaryCommunicationManager
  updater, priority 0). COMMUNICATE is its own action, so a player can't move
  and talk in the same step.
* Episode cap: `max_steps` (maxEpisodeLengthFrames=100).
* Observation: 11x11 egocentric view (5 cells left, right, forward and back,
  centred, forward = up). It is SYMBOLIC one-hot channels rather than the
  substrate's 88x88 RGB, plus the comm bit broadcast as a constant channel.
  Policies trained here therefore can't run in the Lua toy unchanged.

Actions (same indices as toy.py ACTION_SET):
  0 NOOP, 1 FORWARD, 2 BACKWARD, 3 STEP_LEFT, 4 STEP_RIGHT,
  5 TURN_LEFT, 6 TURN_RIGHT, 7 COMMUNICATE

The API mirrors SocialJax/JaxMARL, but uses arrays instead of per-agent dicts:
  obs, state = env.reset(key)              obs: (2, 11, 11, C) float32
  obs, state, reward, done, info = env.step(key, state, actions)  actions: (2,)
step() auto-resets when an episode ends, as SocialJax does.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

ASCII_MAP = [
    "*******      *******",
    "*  a  *      *  u  *",
    "*  _  *      *  _  *",
    "*     *      *     *",
    "*     *      *  l  *",
    "*******      *******",
]
H, W = len(ASCII_MAP), len(ASCII_MAP[0])
_WALL = np.array([[c == "*" for c in row] for row in ASCII_MAP])


def _find(ch):
  return [(x, y) for y, row in enumerate(ASCII_MAP)
          for x, c in enumerate(row) if c == ch]


APPLE_XY = _find("a")[0]
UPPER_XY = _find("u")[0]
LOWER_XY = _find("l")[0]
SPAWNS = sorted(_find("_"))          # [(3,2) apple room, (16,2) pad room]
APPLE_ROOM_SPAWN, PAD_ROOM_SPAWN = SPAWNS

# Orientation 0..3 = N, E, S, W; (dx, dy) with y pointing down.
DIRS = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)

NOOP, FORWARD, BACKWARD, STEP_LEFT, STEP_RIGHT, TURN_LEFT, TURN_RIGHT, COMM = range(8)
NUM_ACTIONS = 8
# Relative move per action: 0 forward, 1 right, 2 back, 3 left; -1 = no move.
_MOVE_REL = jnp.array([-1, 0, 2, 3, 1, -1, -1, -1], dtype=jnp.int32)
_TURN = jnp.array([0, 0, 0, 0, 0, -1, 1, 0], dtype=jnp.int32)

VIEW = 5                              # cells in each direction
VIEW_SIZE = 2 * VIEW + 1
CHANNELS = ("wall", "apple", "upper_pad", "lower_pad", "other_player",
            "out_of_map", "comm_bit")
NUM_CHANNELS = len(CHANNELS)


class State(NamedTuple):
  pos: jnp.ndarray          # (2, 2) int32 x, y
  dir: jnp.ndarray          # (2,) int32
  apple: jnp.ndarray        # () bool, apple exists (upper pad active)
  comm: jnp.ndarray         # () int32, bit observed this step
  t: jnp.ndarray            # () int32 steps taken this episode


class ToyCommsEnv:
  """Binary-communication toy. All methods are pure and jit/vmap friendly."""

  num_agents = 2
  num_actions = NUM_ACTIONS
  obs_shape = (VIEW_SIZE, VIEW_SIZE, NUM_CHANNELS)

  def __init__(self, exists_prob=0.5, reward=1.0, max_steps=100,
               random_orientation=True, fixed_roles=False, comm_enabled=True):
    """fixed_roles=True puts agent 0 in the apple room every episode (sender)
    and agent 1 in the pad room (receiver). This is useful with separate
    networks. Default False matches the substrate (random rooms).
    comm_enabled=False removes the channel: the bit stays 0 and COMMUNICATE
    does nothing. Observation and action spaces are unchanged, so the same
    network trains as a no-communication baseline."""
    self.exists_prob = exists_prob
    self.reward = reward
    self.max_steps = max_steps
    self.random_orientation = random_orientation
    self.fixed_roles = fixed_roles
    self.comm_enabled = comm_enabled
    self._wall = jnp.asarray(_WALL)
    self._spawns = jnp.array(SPAWNS, dtype=jnp.int32)
    self._apple = jnp.array(APPLE_XY, dtype=jnp.int32)
    self._upper = jnp.array(UPPER_XY, dtype=jnp.int32)
    self._lower = jnp.array(LOWER_XY, dtype=jnp.int32)
    # View offsets: row i, col j -> forward = VIEW - i, right = j - VIEW.
    i, j = np.meshgrid(np.arange(VIEW_SIZE), np.arange(VIEW_SIZE), indexing="ij")
    self._fwd = jnp.asarray(VIEW - i)
    self._right = jnp.asarray(j - VIEW)

  # ------------------------------------------------------------------ reset
  def reset(self, key):
    k_room, k_dir, k_apple = jax.random.split(key, 3)
    swap = (jnp.array(False) if self.fixed_roles
            else jax.random.bernoulli(k_room))
    order = jnp.where(swap, jnp.array([1, 0]), jnp.array([0, 1]))
    pos = self._spawns[order]
    if self.random_orientation:
      d = jax.random.randint(k_dir, (2,), 0, 4)
    else:
      d = jnp.zeros((2,), jnp.int32)
    apple = jax.random.uniform(k_apple) < self.exists_prob
    state = State(pos=pos, dir=d.astype(jnp.int32), apple=apple,
                  comm=jnp.int32(0), t=jnp.int32(0))
    return self.get_obs(state), state

  # ------------------------------------------------------------------- step
  def step_env(self, state, actions):
    actions = actions.astype(jnp.int32)
    rel = _MOVE_REL[actions]
    moving = rel >= 0
    abs_dir = (state.dir + jnp.maximum(rel, 0)) % 4
    target = state.pos + DIRS[abs_dir] * moving[:, None]
    tx, ty = target[:, 0], target[:, 1]
    in_map = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)
    blocked = ~in_map | self._wall[jnp.clip(ty, 0, H - 1), jnp.clip(tx, 0, W - 1)]
    # Players are in separate rooms, so they can't collide; checked anyway.
    other_target = target[::-1]
    blocked = blocked | jnp.all(target == other_target, axis=1) & moving
    new_pos = jnp.where(blocked[:, None], state.pos, target)
    new_dir = (state.dir + _TURN[actions]) % 4
    comm = jnp.any(actions == COMM).astype(jnp.int32) * int(self.comm_enabled)

    on_upper = jnp.all(new_pos == self._upper, axis=1) & moving & ~blocked
    on_lower = jnp.all(new_pos == self._lower, axis=1) & moving & ~blocked
    entered = jnp.any(on_upper | on_lower)
    correct = jnp.where(state.apple, jnp.any(on_upper), jnp.any(on_lower))
    r = jnp.where(correct, self.reward, 0.0)
    t = state.t + 1
    done = entered | (t >= self.max_steps)

    new_state = State(pos=new_pos, dir=new_dir, apple=state.apple, comm=comm, t=t)
    info = {
        "episode_done": done,
        "success": correct,
        "pad_entered": entered,
        "episode_length": t,
        "apple": state.apple,
        "comm": comm,
    }
    rewards = jnp.full((2,), r, dtype=jnp.float32)
    return self.get_obs(new_state), new_state, rewards, done, info

  def step(self, key, state, actions):
    obs_st, st, rewards, done, info = self.step_env(state, actions)
    obs_re, st_re = self.reset(key)
    st = jax.tree.map(lambda a, b: jnp.where(done, a, b), st_re, st)
    obs = jnp.where(done, obs_re, obs_st)
    return obs, st, rewards, done, info

  # -------------------------------------------------------------------- obs
  def _layers(self, state):
    """(H, W, C-1) world layers excluding the comm bit and other-player."""
    z = jnp.zeros((H, W), jnp.float32)
    apple = z.at[self._apple[1], self._apple[0]].set(state.apple.astype(jnp.float32))
    upper = z.at[self._upper[1], self._upper[0]].set(1.0)
    lower = z.at[self._lower[1], self._lower[0]].set(1.0)
    return jnp.stack([self._wall.astype(jnp.float32), apple, upper, lower], -1)

  def _view(self, layers, state, a):
    d = state.dir[a]
    f, rgt = DIRS[d], DIRS[(d + 1) % 4]
    wx = state.pos[a, 0] + self._fwd * f[0] + self._right * rgt[0]
    wy = state.pos[a, 1] + self._fwd * f[1] + self._right * rgt[1]
    inside = (wx >= 0) & (wx < W) & (wy >= 0) & (wy < H)
    cx, cy = jnp.clip(wx, 0, W - 1), jnp.clip(wy, 0, H - 1)
    world = layers[cy, cx] * inside[..., None]
    o = state.pos[1 - a]
    other = ((wx == o[0]) & (wy == o[1]) & inside).astype(jnp.float32)
    oob = (~inside).astype(jnp.float32)
    comm = jnp.full((VIEW_SIZE, VIEW_SIZE), state.comm, jnp.float32)
    return jnp.concatenate(
        [world, other[..., None], oob[..., None], comm[..., None]], -1)

  def get_obs(self, state):
    layers = self._layers(state)
    return jnp.stack([self._view(layers, state, a) for a in range(2)])

  # ---------------------------------------------------------------- helpers
  @staticmethod
  def in_apple_room(state):
    """(2,) bool: which agents are in the apple room (x < 7)."""
    return state.pos[:, 0] < 7
