import random
import numpy as np
import pandas as pd
import uuid
# reproducibilidad
SEED = 42
RNG = np.random.RandomState(SEED)
random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# Config global / schema
# -------------------------
SCHEMA = {
    'src_port': {'type': 'int', 'range': (1024, 65535)},
    'dst_port': {'type': 'int', 'range': (1, 65535)},
    'protocol': {'type': 'cat', 'vals': ['TCP', 'UDP', 'ICMP']},
    'packet_count': {'type': 'int', 'range': (1, 2000)},
    'conn_state': {'type': 'cat', 'vals': ['EST', 'SYN', 'FIN', 'RST']},
    'bytes_transferred': {'type': 'float', 'range': (0.0, 2e6)},
    'timestamp': {'type': 'time'}
}
BASE_FEATURES = list(SCHEMA.keys())

NUM_CLASSES = 6
ATTACK_LABELS = {0:'Normal',1:'DoS',2:'Probe',3:'R2L',4:'U2R',5:'Worm'}
ATTACK_TO_ID = {v:k for k,v in ATTACK_LABELS.items()}
ATTACK_PRIORS = {'Normal':0.942,'DoS':0.02,'Probe':0.01,'R2L':0.015,'U2R':0.008,'Worm':0.005}
ATTACK_NAMES = list(ATTACK_PRIORS.keys())
ATTACK_PROBS = list(ATTACK_PRIORS.values())

_PROTO_VALS = np.array(SCHEMA['protocol']['vals'], dtype=object)
_CONN_VALS = np.array(SCHEMA['conn_state']['vals'], dtype=object)


# -------------------------
# Fast categorical sampling (Alias method)
# -------------------------
class _AliasSampler:
    """O(1) discrete sampler after O(K) preprocessing."""

    def __init__(self, probs, rng: np.random.RandomState):
        p = np.asarray(probs, dtype=np.float64)
        if p.ndim != 1:
            raise ValueError("probs must be a 1D array")
        if np.any(p < 0):
            raise ValueError("probs must be non-negative")
        s = float(p.sum())
        if not np.isfinite(s) or s <= 0:
            raise ValueError("probs must sum to a positive finite value")
        p = p / s

        self.rng = rng
        self.K = int(p.size)
        self.prob = np.empty(self.K, dtype=np.float64)
        self.alias = np.empty(self.K, dtype=np.int32)

        scaled = p * self.K
        small = []
        large = []
        for i in range(self.K):
            (small if scaled[i] < 1.0 else large).append(i)

        while small and large:
            sidx = small.pop()
            lidx = large.pop()
            self.prob[sidx] = scaled[sidx]
            self.alias[sidx] = lidx
            scaled[lidx] = (scaled[lidx] + scaled[sidx]) - 1.0
            (small if scaled[lidx] < 1.0 else large).append(lidx)

        for idx in large:
            self.prob[idx] = 1.0
            self.alias[idx] = idx
        for idx in small:
            self.prob[idx] = 1.0
            self.alias[idx] = idx

    def sample_one(self) -> int:
        k = int(self.rng.randint(0, self.K))
        return k if float(self.rng.random()) < float(self.prob[k]) else int(self.alias[k])

    def sample_n(self, n: int) -> np.ndarray:
        n = int(n)
        k = self.rng.randint(0, self.K, size=n)
        u = self.rng.random(size=n)
        return np.where(u < self.prob[k], k, self.alias[k]).astype(np.int32, copy=False)


# -------------------------
# SyntheticTrafficGenerator (refactored, same external API)
# -------------------------
class SyntheticTrafficGenerator:
    """Rule-based traffic generator with explicit drift modes.

    External behavior preserved:
    - Same schema (src_port, dst_port, protocol, packet_count, conn_state, bytes_transferred, timestamp)
    - Same labels (0..5 with ATTACK_LABELS mapping)
    - Same `produce()` output JSON schema
    - Same `generate_dataset()` return shape/columns

    Drift semantics:
    - normal: baseline distributions + deterministic label = sampled attack id
    - data_drift: feature distributions shift; label semantics unchanged
    - concept_drift: features sampled like normal; label assigned by concept oracle (fX_concept)

    Performance:
    - `produce()` is O(1), has no loops and no rejection/correction.
    - `generate_dataset()` is vectorized.
    """

    def __init__(
        self,
        start_ts: str = "2026-01-12 18:00:00",
        epsilon_seconds: int = 60,
        rng: np.random.RandomState = None,
    ):
        self.start_ts = pd.to_datetime(start_ts)
        self.epsilon = int(epsilon_seconds)
        self.rng = rng if rng is not None else np.random.RandomState(None)
        self.n_step = 0

        # Attack sampling is O(1) per sample (no loops in produce).
        self._attack_sampler = _AliasSampler(ATTACK_PROBS, self.rng)

        # Precompute categorical encodings for vectorized paths.
        self._proto_to_idx = {v: i for i, v in enumerate(_PROTO_VALS.tolist())}
        self._conn_to_idx = {v: i for i, v in enumerate(_CONN_VALS.tolist())}

    # -------- Timestamp sequencing (monotonic, block-based) --------
    def next_timestamp(self):
        i = self.n_step
        block_start = self.start_ts + pd.Timedelta(seconds=i * self.epsilon)
        delta = float(self.rng.uniform(0, self.epsilon))
        ts = block_start + pd.Timedelta(seconds=delta)
        self.n_step += 1
        return ts.replace(microsecond=0)

    def next_timestamps(self, n: int) -> pd.DatetimeIndex:
        n = int(n)
        i = self.n_step + np.arange(n, dtype=np.int64)
        base = self.start_ts.to_datetime64() + (i * self.epsilon).astype('timedelta64[s]')
        delta = (self.rng.uniform(0, self.epsilon, size=n)).astype(np.int64)
        ts = base + delta.astype('timedelta64[s]')
        self.n_step += n
        return pd.to_datetime(ts).floor('S')

    def reset(self, start_ts: str = None, epsilon_seconds: int = None):
        if start_ts is not None:
            self.start_ts = pd.to_datetime(start_ts)
        if epsilon_seconds is not None:
            self.epsilon = int(epsilon_seconds)
        self.n_step = 0

    # -------- Latent structure helpers (explicit, human interpretable) --------
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def cat_to_num(val, choices):
        idx = choices.index(val)
        return -1 + 2 * idx / (len(choices) - 1) if len(choices) > 1 else 0.0

    # -------- Oracle logic (kept compatible; used for concept drift + evaluation) --------
    def fX_normal(self, row):
        pc = float(row['packet_count']) / 2000.0
        bt = float(row['bytes_transferred']) / 2e6
        sp = float(row['src_port']) / 65535.0
        dp = float(row['dst_port']) / 65535.0
        proto = self.cat_to_num(row['protocol'], SCHEMA['protocol']['vals'])
        state = self.cat_to_num(row['conn_state'], SCHEMA['conn_state']['vals'])
        score = np.zeros(NUM_CLASSES)
        score[1] += 3.0 * pc**2 + 1.5 * self.sigmoid(proto - 0.5) + 0.5 * (1 - sp) + 0.3 * pc * proto
        score[2] += 1.5 * np.isin(row['dst_port'], [21, 22, 23, 80, 443]) + 2.0 * (pc * (1 - pc)) + 0.5 * dp * pc + 0.3 * dp * state
        score[3] += 2.0 * state * (1 - bt) + 0.5 * (1 - dp) + 0.3 * bt * state + 1.0 * (1 - pc) * state
        score[4] += 1.5 * bt**2 + 0.5 * state * sp + 0.3 * bt * pc + 0.5 * sp * (1 - dp)
        score[5] += 2.0 * bt * pc + 0.5 * (1 * (row['protocol'] != 'ICMP')) + 0.3 * sp * dp
        score[0] += 1.0 - abs(pc - 0.3) - abs(bt - 0.3) + 0.2 * (1 - abs(proto)) + 0.1 * (1 - abs(state))
        score += 0.02 * self.rng.randn(NUM_CLASSES)
        return int(np.argmax(score))

    def fX_concept(self, row):
        pc = float(row['packet_count']) / 2000.0
        bt = float(row['bytes_transferred']) / 2e6
        sp = float(row['src_port']) / 65535.0
        dp = float(row['dst_port']) / 65535.0
        proto = self.cat_to_num(row['protocol'], SCHEMA['protocol']['vals'])
        state = self.cat_to_num(row['conn_state'], SCHEMA['conn_state']['vals'])
        score = np.zeros(NUM_CLASSES)
        score[1] += 4.0 * bt + 0.8 * (row['protocol'] == 'ICMP') - 0.3 * pc
        src_mod = (row['src_port'] % 1000) / 1000.0
        score[2] += 3.0 * (src_mod < 0.2) + 1.0 * np.isin(row['dst_port'], [21, 22, 23, 80, 443])
        score[3] += 2.0 * (row['protocol'] == 'ICMP') + 1.0 * (0.05 < bt < 0.2)
        score[4] += 2.0 * (pc > 0.7) + 1.2 * bt
        score[5] += 1.5 * sp * dp + 0.7 * (0.2 < pc < 0.6)
        score[0] += 0.8 - 0.2 * (abs(pc - 0.35) + abs(bt - 0.35))
        score += 0.05 * self.rng.randn(NUM_CLASSES)
        return int(np.argmax(score))

    # -------- Feature sampling (attack semantics + drift; no rejection) --------
    def _sample_attack_id_one(self) -> int:
        return int(self._attack_sampler.sample_one())

    def _sample_attack_ids(self, n: int) -> np.ndarray:
        return self._attack_sampler.sample_n(n)

    def _baseline_features_one(self, trend: str):
        if trend in ('normal', 'concept_drift'):
            src_port = int(self.rng.randint(1024, 65535))
            dst_port = int(self.rng.randint(1, 65535))
            proto = str(self.rng.choice(['TCP', 'UDP']))
            packet_count = int(self.rng.randint(1, 400))
            conn_state = 'EST'
            bytes_transferred = float(self.rng.uniform(1e3, 5e5))
        else:  # data_drift
            # Moderate covariate shift: move normals towards heavier traffic and noisier states
            # but keep attack semantics unchanged (labels remain coherent).
            src_port = int(1024 + (45000 - 1024) * float(self.rng.beta(a=2, b=4)))
            dst_port = int(1 + (45000 - 1) * float(self.rng.beta(a=2, b=4)))
            proto = str(self.rng.choice(['ICMP', 'TCP', 'UDP'], p=[0.25, 0.40, 0.35]))
            packet_count = int(50 + (1200 - 50) * float(self.rng.beta(a=2, b=5)))
            conn_state = str(self.rng.choice(['EST', 'SYN', 'RST'], p=[0.55, 0.35, 0.10]))
            bytes_transferred = float(1e4 + (1.2e6 - 1e4) * float(self.rng.beta(a=2, b=5)))
        return src_port, dst_port, proto, packet_count, conn_state, bytes_transferred

    def _apply_attack_semantics_one(
        self,
        attack_id: int,
        src_port: int,
        dst_port: int,
        proto: str,
        packet_count: int,
        conn_state: str,
        bytes_transferred: float,
    ):
        # Deterministic-by-construction attack regions (no oracle correction).
        if attack_id == ATTACK_TO_ID['DoS']:
            packet_count = int(self.rng.randint(1200, 2000))
            proto = str(self.rng.choice(['ICMP', 'UDP']))
        elif attack_id == ATTACK_TO_ID['Probe']:
            dst_port = int(self.rng.choice([21, 22, 23, 80, 443]))
            packet_count = int(self.rng.randint(200, 800))
        elif attack_id == ATTACK_TO_ID['R2L']:
            conn_state = 'RST'
            bytes_transferred = float(self.rng.uniform(0, 200))
        elif attack_id == ATTACK_TO_ID['U2R']:
            bytes_transferred = float(self.rng.uniform(8e5, 1.3e6))
            conn_state = 'EST'
        elif attack_id == ATTACK_TO_ID['Worm']:
            bytes_transferred = float(self.rng.uniform(1.5e6, 2e6))
            packet_count = int(self.rng.randint(600, 1400))
        # Normal: keep baseline.
        return src_port, dst_port, proto, packet_count, conn_state, bytes_transferred

    def _sample_row_one(self, attack_id: int, trend: str):
        src_port, dst_port, proto, packet_count, conn_state, bytes_transferred = self._baseline_features_one(trend)
        src_port, dst_port, proto, packet_count, conn_state, bytes_transferred = self._apply_attack_semantics_one(
            attack_id, src_port, dst_port, proto, packet_count, conn_state, bytes_transferred
        )
        if trend == 'data_drift':
            packet_count = int(np.clip(packet_count + float(self.rng.normal(0, 80)), 1, 2000))
            bytes_transferred = float(np.clip(bytes_transferred * float(self.rng.lognormal(mean=0.0, sigma=0.20)), 0.0, 2e6))
        row = {
            'src_port': int(src_port),
            'dst_port': int(dst_port),
            'protocol': proto,
            'packet_count': int(packet_count),
            'conn_state': conn_state,
            'bytes_transferred': float(bytes_transferred),
            'timestamp': self.next_timestamp(),
        }
        return row

    # -------- Public single-sample API (fast, O(1), loop-free) --------
    def produce(self, trend: str = 'normal'):
        """Generate exactly one JSON-friendly sample.

        Requirements met:
        - No loops
        - No rejection sampling
        - No oracle correction
        - Only: sample features, deterministic label, timestamp, format output
        """
        attack_id = self._sample_attack_id_one()
        row = self._sample_row_one(attack_id, trend)

        if trend == 'concept_drift':
            label = self.fX_concept(row)
        else:
            label = int(attack_id)

        return {
            "timestamp": row["timestamp"].isoformat(), #quitarlo y en producer para que el de kafka sea el que se queda
            "event_id": str(uuid.uuid4()),
            "properties": {
                "src_port": int(row["src_port"]),
                "dst_port": int(row["dst_port"]),
                "protocol": row["protocol"],
                "packet_count": int(row["packet_count"]),
                "conn_state": row["conn_state"],
                "bytes_transferred": float(row["bytes_transferred"]),
            },
            "label": int(label),
        }

    # -------- Vectorized oracles for batch (no Python loops over rows) --------
    def _fX_concept_vec(
        self,
        src_port: np.ndarray,
        dst_port: np.ndarray,
        proto_idx: np.ndarray,
        packet_count: np.ndarray,
        conn_idx: np.ndarray,
        bytes_transferred: np.ndarray,
    ) -> np.ndarray:
        pc = packet_count.astype(np.float64) / 2000.0
        bt = bytes_transferred.astype(np.float64) / 2e6
        sp = src_port.astype(np.float64) / 65535.0
        dp = dst_port.astype(np.float64) / 65535.0

        is_icmp = (proto_idx == int(self._proto_to_idx['ICMP']))

        score = np.zeros((pc.size, NUM_CLASSES), dtype=np.float64)
        score[:, 1] += 4.0 * bt + 0.8 * is_icmp.astype(np.float64) - 0.3 * pc
        src_mod = (src_port % 1000).astype(np.float64) / 1000.0
        score[:, 2] += 3.0 * (src_mod < 0.2).astype(np.float64) + 1.0 * np.isin(dst_port, [21, 22, 23, 80, 443]).astype(np.float64)
        score[:, 3] += 2.0 * is_icmp.astype(np.float64) + 1.0 * ((bt > 0.05) & (bt < 0.2)).astype(np.float64)
        score[:, 4] += 2.0 * (pc > 0.7).astype(np.float64) + 1.2 * bt
        score[:, 5] += 1.5 * sp * dp + 0.7 * ((pc > 0.2) & (pc < 0.6)).astype(np.float64)
        score[:, 0] += 0.8 - 0.2 * (np.abs(pc - 0.35) + np.abs(bt - 0.35))
        score += 0.05 * self.rng.randn(pc.size, NUM_CLASSES)
        return np.argmax(score, axis=1).astype(np.int32, copy=False)

    # -------- Batch generation (vectorized; shares same distributions/semantics) --------
    def generate_dataset(self, n: int, trend: str = 'normal'):
        """Generate a pandas DataFrame with an `attack` column.

        Vectorized generation:
        - Samples all attacks in bulk.
        - Samples timestamps in bulk.
        - Applies attack semantics via boolean masks.
        """
        n = int(n)
        attack_id = self._sample_attack_ids(n)
        ts = self.next_timestamps(n)

        # Baseline distributions
        if trend in ('normal', 'concept_drift'):
            src_port = self.rng.randint(1024, 65535, size=n).astype(np.int32)
            dst_port = self.rng.randint(1, 65535, size=n).astype(np.int32)
            proto_idx = self.rng.choice(
                np.array([self._proto_to_idx['TCP'], self._proto_to_idx['UDP']], dtype=np.int32),
                size=n,
            ).astype(np.int32)
            packet_count = self.rng.randint(1, 400, size=n).astype(np.int32)
            conn_idx = np.full(n, self._conn_to_idx['EST'], dtype=np.int32)
            bytes_transferred = self.rng.uniform(1e3, 5e5, size=n).astype(np.float64)
        else:  # data_drift
            # Moderate covariate shift (aim: val_data_drift accuracy ~0.8-0.9).
            src_port = (1024 + (45000 - 1024) * self.rng.beta(a=2, b=4, size=n)).astype(np.int32)
            dst_port = (1 + (45000 - 1) * self.rng.beta(a=2, b=4, size=n)).astype(np.int32)
            proto_idx = self.rng.choice(
                np.array([self._proto_to_idx['ICMP'], self._proto_to_idx['TCP'], self._proto_to_idx['UDP']], dtype=np.int32),
                p=[0.25, 0.40, 0.35],
                size=n,
            ).astype(np.int32)
            packet_count = (50 + (1200 - 50) * self.rng.beta(a=2, b=5, size=n)).astype(np.int32)
            conn_idx = self.rng.choice(
                np.array([self._conn_to_idx['EST'], self._conn_to_idx['SYN'], self._conn_to_idx['RST']], dtype=np.int32),
                p=[0.55, 0.35, 0.10],
                size=n,
            ).astype(np.int32)
            bytes_transferred = (1e4 + (1.2e6 - 1e4) * self.rng.beta(a=2, b=5, size=n)).astype(np.float64)

        # Attack masks
        m_dos = (attack_id == ATTACK_TO_ID['DoS'])
        m_probe = (attack_id == ATTACK_TO_ID['Probe'])
        m_r2l = (attack_id == ATTACK_TO_ID['R2L'])
        m_u2r = (attack_id == ATTACK_TO_ID['U2R'])
        m_worm = (attack_id == ATTACK_TO_ID['Worm'])

        # Apply attack semantics (vectorized; no rejection)
        if np.any(m_dos):
            k = int(m_dos.sum())
            packet_count[m_dos] = self.rng.randint(1200, 2000, size=k)
            proto_idx[m_dos] = self.rng.choice(
                np.array([self._proto_to_idx['ICMP'], self._proto_to_idx['UDP']], dtype=np.int32),
                size=k,
            )
        if np.any(m_probe):
            k = int(m_probe.sum())
            dst_port[m_probe] = self.rng.choice(np.array([21, 22, 23, 80, 443], dtype=np.int32), size=k)
            packet_count[m_probe] = self.rng.randint(200, 800, size=k)
        if np.any(m_r2l):
            k = int(m_r2l.sum())
            conn_idx[m_r2l] = self._conn_to_idx['RST']
            bytes_transferred[m_r2l] = self.rng.uniform(0, 200, size=k)
        if np.any(m_u2r):
            k = int(m_u2r.sum())
            conn_idx[m_u2r] = self._conn_to_idx['EST']
            bytes_transferred[m_u2r] = self.rng.uniform(8e5, 1.3e6, size=k)
        if np.any(m_worm):
            k = int(m_worm.sum())
            bytes_transferred[m_worm] = self.rng.uniform(1.5e6, 2e6, size=k)
            packet_count[m_worm] = self.rng.randint(600, 1400, size=k)
        if trend == 'data_drift':
            packet_count = np.clip(packet_count.astype(np.float64) + self.rng.normal(0, 80, size=n), 1, 2000).astype(np.int32)
            bytes_transferred = np.clip(bytes_transferred * self.rng.lognormal(mean=0.0, sigma=0.20, size=n), 0.0, 2e6).astype(np.float64)

        # Decode categoricals
        protocol = _PROTO_VALS[proto_idx]
        conn_state = _CONN_VALS[conn_idx]

        df = pd.DataFrame({
            'src_port': src_port.astype(int),
            'dst_port': dst_port.astype(int),
            'protocol': protocol,
            'packet_count': packet_count.astype(int),
            'conn_state': conn_state,
            'bytes_transferred': bytes_transferred.astype(float),
            'timestamp': ts,
        })

        if trend == 'concept_drift':
            df['attack'] = self._fX_concept_vec(
                src_port=src_port,
                dst_port=dst_port,
                proto_idx=proto_idx,
                packet_count=packet_count,
                conn_idx=conn_idx,
                bytes_transferred=bytes_transferred,
            ).astype(int)
        else:
            df['attack'] = attack_id.astype(int)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    # Optional explicit batch API for streaming simulators
    def produce_batch(self, n: int, trend: str = 'normal'):
        """Return a list of JSON-friendly dicts (like `produce()`) for integration ease."""
        df = self.generate_dataset(n, trend=trend)
        out = []
        # This loop is intentionally outside `produce()`; batch callers can stream/serialize.
        for r in df.itertuples(index=False):
            out.append({
                "timestamp": pd.Timestamp(r.timestamp).isoformat(),
                "properties": {
                    "src_port": int(r.src_port),
                    "dst_port": int(r.dst_port),
                    "protocol": str(r.protocol),
                    "packet_count": int(r.packet_count),
                    "conn_state": str(r.conn_state),
                    "bytes_transferred": float(r.bytes_transferred),
                },
                "label": int(r.attack),
            })
        return out

def main():
    
    pass


# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    main()

    # if you want to try the pyspark pipeline, uncomment next lines (requires pyspark installed/configured):
    # from pyspark.sql import SparkSession
    # spark = SparkSession.builder.master("local[4]").appName("synth").getOrCreate()
    # df_train_spark = spark.read.parquet("out/train_proc.parquet")
    # df_train_spark.show(5)
    # spark.stop()
