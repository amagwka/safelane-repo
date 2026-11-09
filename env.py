"""
Safety-constrained highway lane-change environment (SafeLane-v0).

A deliberately small, fully reproducible simulator written in numpy so that
every number in the Results section can be regenerated from a seed alone.

Underlying dynamics are continuous; the agent observes a discretised state,
which is what creates genuine partial observability and makes the coverage
of the offline dataset uneven.

Reward and cost are separate signals (constrained MDP formulation):
  reward r  = forward progress - comfort penalty - collision penalty
  cost   c  = 1 if the step is unsafe (collision or sub-threshold headway)
"""

import numpy as np

# ---------------------------------------------------------------- constants
N_LANES = 3
ROAD_LEN = 520.0          # m, ring road
DT = 0.4                  # s per step
HORIZON = 200             # steps per episode
N_TRAFFIC = 12            # other vehicles

V_MIN, V_MAX = 8.0, 32.0  # m/s ego speed limits
ACC = 2.5                 # m/s^2 per accel/decel action
CAR_LEN = 5.0             # m

# safety thresholds
TTC_UNSAFE = 1.6          # s, time-to-collision below this is an unsafe step
GAP_CRASH = CAR_LEN       # m, bumper contact

# discretisation
SPEED_EDGES = [14.0, 21.0, 27.0]           # -> 4 buckets
GAP_EDGES = [12.0, 30.0, 70.0]             # -> 4 buckets
SIDE_EDGES = [15.0, 40.0]                  # -> 3 buckets (blocked/tight/free)

N_SPEED = len(SPEED_EDGES) + 1
N_GAP = len(GAP_EDGES) + 1
N_SIDE = len(SIDE_EDGES) + 1
N_STATES = N_LANES * N_SPEED * N_GAP * N_SIDE * N_SIDE   # 3*4*4*3*3 = 432

# actions
A_KEEP, A_LEFT, A_RIGHT, A_ACC, A_DEC = range(5)
N_ACTIONS = 5
ACTION_NAMES = ["keep", "left", "right", "accel", "decel"]


def _bucket(x, edges):
    b = 0
    for e in edges:
        if x >= e:
            b += 1
    return b


class SafeLane:
    """Ring-road lane-change task with a separate safety-cost signal."""

    def __init__(self, seed=0, obs_noise=0.0):
        self.rng = np.random.default_rng(seed)
        self.obs_noise = float(obs_noise)

    # ------------------------------------------------------------ dynamics
    def reset(self):
        self.t = 0
        self.ego_pos = 0.0
        self.ego_lane = 1
        self.ego_v = 18.0
        # traffic: positions spread over the road, slower than ego top speed
        self.tr_pos = self.rng.uniform(20.0, ROAD_LEN, size=N_TRAFFIC)
        self.tr_lane = self.rng.integers(0, N_LANES, size=N_TRAFFIC)
        self.tr_v = self.rng.uniform(10.0, 24.0, size=N_TRAFFIC)
        self.last_event = "none"
        return self.observe()

    def _gap_ahead(self, lane, pos):
        """Signed distance to the rear bumper of the next car ahead in `lane`."""
        m = self.tr_lane == lane
        if not np.any(m):
            return ROAD_LEN, 0.0
        d = (self.tr_pos[m] - pos) % ROAD_LEN
        i = int(np.argmin(d))
        return float(d[i]), float(self.tr_v[m][i])

    def _gap_behind(self, lane, pos):
        m = self.tr_lane == lane
        if not np.any(m):
            return ROAD_LEN
        d = (pos - self.tr_pos[m]) % ROAD_LEN
        return float(np.min(d))

    def _true_features(self):
        g_own, v_own = self._gap_ahead(self.ego_lane, self.ego_pos)
        sides = {}
        for name, lane in (("left", self.ego_lane - 1), ("right", self.ego_lane + 1)):
            if lane < 0 or lane >= N_LANES:
                sides[name] = -1.0            # no such lane
            else:
                ga, _ = self._gap_ahead(lane, self.ego_pos)
                gb = self._gap_behind(lane, self.ego_pos)
                sides[name] = min(ga, gb)     # tightest side clearance
        return g_own, v_own, sides["left"], sides["right"]

    def _refresh_obs(self):
        """
        Draw ONE sensor reading for this timestep and cache it, so that every
        policy acting at this step sees the same (possibly corrupted) values.
        With obs_noise == 0 this is exactly the true feature vector.
        """
        self._obs_feat = self._noisy_features()
        return self._obs_feat

    def _noisy_features(self):
        g_own, v_own, g_left, g_right = self._true_features()
        if self.obs_noise <= 0.0:
            return g_own, v_own, g_left, g_right, self.ego_v
        s = self.obs_noise
        g_own = max(0.0, g_own * (1.0 + self.rng.normal(0.0, s)))
        if g_left >= 0.0:
            g_left = max(0.0, g_left * (1.0 + self.rng.normal(0.0, s)))
        if g_right >= 0.0:
            g_right = max(0.0, g_right * (1.0 + self.rng.normal(0.0, s)))
        v_obs = self.ego_v + self.rng.normal(0.0, 2.0 * s)
        return g_own, v_own, g_left, g_right, v_obs

    def observe(self):
        """Discretised observation index built from the cached sensor reading."""
        g_own, v_own, g_left, g_right, v_obs = self._refresh_obs()

        b_lane = self.ego_lane
        b_speed = _bucket(v_obs, SPEED_EDGES)
        b_gap = _bucket(g_own, GAP_EDGES)
        b_left = 0 if g_left < 0.0 else _bucket(g_left, SIDE_EDGES)
        b_right = 0 if g_right < 0.0 else _bucket(g_right, SIDE_EDGES)

        return (((b_lane * N_SPEED + b_speed) * N_GAP + b_gap) * N_SIDE + b_left) * N_SIDE + b_right

    def step(self, a):
        """Returns (obs, reward, cost, done, info)."""
        self.t += 1
        comfort = 0.0
        self.last_event = "none"

        # lateral action
        if a == A_LEFT:
            comfort = 0.25
            tgt = self.ego_lane - 1
            if tgt >= 0:
                if self._lane_change_blocked(tgt):
                    self.last_event = "unsafe_cut_in"
                self.ego_lane = tgt
            else:
                self.last_event = "invalid_lane"
                comfort = 0.6          # steering into the barrier
        elif a == A_RIGHT:
            comfort = 0.25
            tgt = self.ego_lane + 1
            if tgt < N_LANES:
                if self._lane_change_blocked(tgt):
                    self.last_event = "unsafe_cut_in"
                self.ego_lane = tgt
            else:
                self.last_event = "invalid_lane"
                comfort = 0.6
        elif a == A_ACC:
            self.ego_v = min(V_MAX, self.ego_v + ACC * DT)
        elif a == A_DEC:
            self.ego_v = max(V_MIN, self.ego_v - ACC * DT)

        # longitudinal integration
        self.ego_pos = (self.ego_pos + self.ego_v * DT) % ROAD_LEN
        self.tr_pos = (self.tr_pos + self.tr_v * DT) % ROAD_LEN

        # safety evaluation after the move
        g_own, v_own, _, _ = self._true_features()
        closing = self.ego_v - v_own
        ttc = (g_own - CAR_LEN) / closing if closing > 0.1 else np.inf

        collision = g_own <= GAP_CRASH
        unsafe = collision or ttc < TTC_UNSAFE
        cost = 1.0 if unsafe else 0.0

        reward = (self.ego_v * DT) / (V_MAX * DT)       # normalised progress in [0,1]
        reward -= comfort
        if collision:
            reward -= 8.0
            self.last_event = "collision"
            self.ego_v = max(V_MIN, 0.5 * self.ego_v)   # hard brake after contact
        elif unsafe and self.last_event == "none":
            self.last_event = "near_miss"

        done = collision or self.t >= HORIZON
        info = {"event": self.last_event, "ttc": ttc, "collision": collision}
        return self.observe(), reward, cost, done, info

    def _lane_change_blocked(self, lane):
        ga, _ = self._gap_ahead(lane, self.ego_pos)
        gb = self._gap_behind(lane, self.ego_pos)
        return min(ga, gb) < 14.0


# ------------------------------------------------------- behaviour policy
def behaviour_action(env, rng, eps=0.25):
    """
    Hand-coded, deliberately mediocre driver used to generate the offline
    dataset: it is cautious about the car ahead but ignores side traffic,
    so the logged data contains both useful and unsafe behaviour.
    """
    if rng.random() < eps:
        return int(rng.integers(0, N_ACTIONS))

    # read the same cached sensor values the learned policies see, so that the
    # noise sweep degrades every policy equally
    g_own, v_own, g_left, g_right, v_obs = env._obs_feat
    closing = v_obs - v_own
    ttc = (g_own - CAR_LEN) / closing if closing > 0.1 else np.inf

    if ttc < 3.2 or g_own < 18.0:
        # try to escape sideways, preferring the roomier side, else brake
        if g_left > 22.0 and g_left >= g_right:
            return A_LEFT
        if g_right > 22.0:
            return A_RIGHT
        return A_DEC
    if v_obs < 24.0 and g_own > 45.0:
        return A_ACC
    return A_KEEP
