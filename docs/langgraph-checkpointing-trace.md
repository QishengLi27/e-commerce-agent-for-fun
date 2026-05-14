# LangGraph Checkpointing & Persistence — Line-by-Line

This traces every file and every method in the checkpointing system, from your code to disk and back.

---

## The Big Picture

```
Your code:  graph.invoke(input, config={"configurable": {"thread_id": "t1"}})
                                                                    │
                              ┌─────────────────────────────────────┘
                              ▼
                    Pregel.stream()                     [pregel/main.py:2491]
                              │
                              ▼
                    SyncPregelLoop.__enter__()          [pregel/_loop.py:1223]
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
        checkpointer.get_tuple()    empty_checkpoint()
        [checkpoint/memory:135]     [pregel/_checkpoint.py:16]
        (loads previous state)      (first-run fallback)
                    │                    │
                    └─────────┬──────────┘
                              ▼
                    channels_from_checkpoint()         [pregel/_checkpoint.py:58]
                    (restores channel objects from saved values)
                              │
                              ▼
                    Loop: tick() → after_tick()
                              │
                    after_tick calls _put_checkpoint()  [pregel/_loop.py:872]
                              │
                              ▼
                    checkpointer.put()                 [checkpoint/memory:326]
                    (saves state to memory/DB)
```

---

# Part 1: The Checkpointer Interface

**File:** `langgraph/checkpoint/base/__init__.py`

## The core types

### Checkpoint — what's saved

```python
class Checkpoint(TypedDict):
    v: int                        # format version (currently 4)
    id: str                       # unique, monotonically increasing ID
    ts: str                       # ISO 8601 timestamp
    channel_values: dict[str, Any] # channel_name → deserialized value
    channel_versions: dict[str, str | int | float]  # channel_name → version
    versions_seen: dict            # INTERRUPT → {channel: last_seen_version}
    updated_channels: list[str] | None  # which channels changed this step
```

**Key insight:** `channel_values` is a partial snapshot. Only channels that have been written to appear here. On restore, missing channels get their default values.

### CheckpointMetadata — about the checkpoint

```python
class CheckpointMetadata(TypedDict, total=False):
    source: Literal["input", "loop", "update", "fork"]
    step: int                           # -2=empty, -1=input, 0=first loop, ...
    parents: dict[str, str]             # namespace → checkpoint_id
    run_id: str                         # which run created this
```

The `source` field tells you WHY the checkpoint was created:
- `"input"`: user just provided input (before first node runs)
- `"loop"`: after each super-step of the Pregel loop
- `"update"`: manual state update via `graph.update_state()`
- `"fork"`: time-travel — re-execution from a historical checkpoint

### CheckpointTuple — what get_tuple() returns

```python
class CheckpointTuple(NamedTuple):
    config: RunnableConfig                # config to get back here
    checkpoint: Checkpoint                # the actual state snapshot
    metadata: CheckpointMetadata          # step, source, parents
    parent_config: RunnableConfig | None  # config for the parent checkpoint
    pending_writes: list[PendingWrite]    # writes not yet applied to state
```

### The BaseCheckpointSaver interface

Every checkpointer (InMemorySaver, PostgresSaver, etc.) must implement:

| Method | Purpose |
|--------|---------|
| `get_tuple(config)` | Load the latest checkpoint for a thread |
| `put(config, checkpoint, metadata, new_versions)` | Save a checkpoint |
| `put_writes(config, writes, task_id)` | Save pending writes mid-step |
| `list(config, filter, before, limit)` | List checkpoints (for time-travel) |
| `get_next_version(current, channel)` | Monotonically increasing version number |
| `delete_thread(thread_id)` | Delete all state for a thread |

---

# Part 2: MemorySaver (InMemorySaver) — The In-Memory Implementation

**File:** `langgraph/checkpoint/memory/__init__.py`

This is the simplest checkpointer. It stores everything in `defaultdict` structures in memory.

## Storage layout

```python
class InMemorySaver:
    # thread_id → checkpoint_ns → checkpoint_id → (checkpoint_b, metadata_b, parent_id)
    storage: defaultdict[str, dict[str, dict[str, tuple[bytes, bytes, str | None]]]]

    # (thread_id, checkpoint_ns, checkpoint_id) → (task_id, write_idx) → write_data
    writes: defaultdict[tuple, dict[tuple, tuple[str, str, bytes, str]]]

    # (thread_id, checkpoint_ns, channel, version) → (serializer_type, bytes)
    blobs: dict[tuple, tuple[str, bytes]]
```

**Key design:** channel values are stored separately in `blobs`, keyed by `(thread_id, checkpoint_ns, channel, version)`. This allows sharing unchanged values across checkpoints — if channel X hasn't changed between checkpoint 1 and checkpoint 2, they both point to the same blob.

## get_tuple() — loading a checkpoint

```python
def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
    thread_id = config["configurable"]["thread_id"]
    checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

    if checkpoint_id := get_checkpoint_id(config):
        # Explicit checkpoint ID — fetch that exact one (time-travel)
        saved = self.storage[thread_id][checkpoint_ns].get(checkpoint_id)
    else:
        # No ID — fetch the LATEST checkpoint for this thread
        if checkpoints := self.storage[thread_id][checkpoint_ns]:
            checkpoint_id = max(checkpoints.keys())
            saved = checkpoints[checkpoint_id]

    if saved:
        checkpoint, metadata, parent_checkpoint_id = saved

        # 1. Deserialize the checkpoint structure
        checkpoint_ = self.serde.loads_typed(checkpoint)

        # 2. Load channel VALUES from blobs
        channel_values = self._load_blobs(thread_id, checkpoint_ns, checkpoint_["channel_versions"])

        # 3. Load pending writes
        writes = self.writes[(thread_id, checkpoint_ns, checkpoint_id)].values()

        return CheckpointTuple(
            config=...,
            checkpoint={**checkpoint_, "channel_values": channel_values},
            metadata=self.serde.loads_typed(metadata),
            pending_writes=[(id, c, self.serde.loads_typed(v)) for id, c, v, _ in writes],
            parent_config=...,
        )
    return None  # First invocation — no checkpoint exists
```

**Step by step:**
1. Get `thread_id` from config
2. Look up `storage[thread_id][checkpoint_ns]` → dict of `checkpoint_id → saved_data`
3. If no `checkpoint_id` requested, take `max(checkpoints.keys())` (latest)
4. Deserialize the checkpoint metadata
5. For each channel version, load the actual value from `blobs`
6. Load pending writes that haven't been applied
7. Return a `CheckpointTuple`

## put() — saving a checkpoint

```python
def put(self, config, checkpoint, metadata, new_versions):
    # 1. Separate channel values from checkpoint structure
    c = checkpoint.copy()
    thread_id = config["configurable"]["thread_id"]
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
    values = c.pop("channel_values")

    # 2. Save each channel value as a blob
    for k, v in new_versions.items():
        self.blobs[(thread_id, checkpoint_ns, k, v)] = self.serde.dumps_typed(values[k])

    # 3. Save checkpoint structure (without channel values)
    self.storage[thread_id][checkpoint_ns][checkpoint["id"]] = (
        self.serde.dumps_typed(c),           # checkpoint sans values
        self.serde.dumps_typed(metadata),    # metadata
        config["configurable"].get("checkpoint_id"),  # parent checkpoint ID
    )
```

**Key design:** The checkpoint structure (id, ts, channel_versions, updated_channels) is stored separately from channel values. This is the blob/cell pattern — the checkpoint acts as a pointer to channel values, not a container of them.

---

# Part 3: The Pregel Loop — Where Checkpointing Happens

**File:** `langgraph/pregel/_loop.py`

The `PregelLoop` class manages the checkpoint lifecycle during execution.

## Phase 1: `__enter__` — Load or create the first checkpoint

```python
# SyncPregelLoop.__enter__() — line 1223

def __enter__(self):
    if not self.checkpointer:
        saved = None                           # No checkpointer → fresh start
    elif self.checkpoint_config[CONF].get(CONFIG_KEY_CHECKPOINT_ID):
        saved = self.checkpointer.get_tuple(self.checkpoint_config)  # Exact replay
    else:
        saved = self.checkpointer.get_tuple(self.checkpoint_config)  # Latest

    if saved is None:
        # First-ever invocation — create empty checkpoint
        saved = CheckpointTuple(
            self.checkpoint_config,
            empty_checkpoint(),           # v=4, id=uuid, channel_values={}, ...
            {"step": -2},                 # step=-2 means "never run"
            None,
            [],
        )

    # Restore channel objects from saved values
    self.checkpoint = saved.checkpoint
    self.checkpoint_pending_writes = [
        (str(tid), k, v) for tid, k, v in saved.pending_writes
    ]
    self.channels, self.managed = channels_from_checkpoint(self.specs, self.checkpoint)

    # Set up step counter
    self.step = self.checkpoint_metadata["step"] + 1
    self.stop = self.step + self.config["recursion_limit"] + 1

    # Apply input to channels (creates "input" checkpoint)
    self.updated_channels = self._first(input_keys=self.input_keys, ...)
```

**Step-by-step:**

1. **No checkpointer?** → Fresh start every time, no persistence
2. **Checkpoint ID in config?** → Load that exact checkpoint (time-travel / replay)
3. **Otherwise?** → Load the latest checkpoint for this thread
4. **No saved checkpoint?** → Create an `empty_checkpoint()` with `step=-2`
5. **Restore channels:** Call `channels_from_checkpoint()` which calls `channel.from_checkpoint(saved_value)` for each channel in the spec
6. **Apply input:** Calls `_first()` which maps the user's input into channel writes and creates the `"input"` checkpoint

## Phase 2: `tick()` — Execute one super-step

```python
def tick(self):
    # Check iteration limit
    if self.step > self.stop:
        self.status = "out_of_steps"
        return False

    # Prepare next tasks (which nodes to run)
    self.tasks = prepare_next_tasks(
        self.checkpoint, self.checkpoint_pending_writes, self.nodes,
        self.channels, self.managed, ...
    )

    # If no tasks, done
    if not self.tasks:
        self.status = "done"
        return False

    # Check interrupt_before
    if self.interrupt_before and should_interrupt(...):
        self.status = "interrupt_before"
        raise GraphInterrupt()

    return True  # ← nodes must still be executed (happens outside tick)
```

## Phase 3: `after_tick()` — Apply writes and save checkpoint

```python
def after_tick(self):
    # 1. Collect writes from all tasks
    writes = [w for t in self.tasks.values() for w in t.writes]

    # 2. Apply writes to channels (this updates channel versions)
    self.updated_channels = apply_writes(
        self.checkpoint, self.channels, self.tasks.values(),
        self.checkpointer_get_next_version, ...
    )

    # 3. Emit values output
    self._emit("values", map_output_values, self.output_keys, writes, self.channels)

    # 4. Clear pending writes
    self.checkpoint_pending_writes.clear()

    # 5. SAVE CHECKPOINT
    self._put_checkpoint({"source": "loop"})

    # 6. Check interrupt_after
    if self.interrupt_after and should_interrupt(...):
        self.status = "interrupt_after"
        raise GraphInterrupt()
```

## Phase 4: `_put_checkpoint()` — The actual save

```python
def _put_checkpoint(self, metadata: CheckpointMetadata):
    # 1. Skip if checkpoint ID hasn't changed
    exiting = metadata is self.checkpoint_metadata
    if exiting and self.checkpoint["id"] == self.checkpoint_id_saved:
        return

    # 2. Set step and parents in metadata
    if not exiting:
        metadata["step"] = self.step
        metadata["parents"] = self.config[CONF].get(CONFIG_KEY_CHECKPOINT_MAP, {})
        self.checkpoint_metadata = metadata

    # 3. Should we actually persist?
    do_checkpoint = (
        self._checkpointer_put_after_previous is not None
        and (exiting or self.durability != "exit")
    )

    # 4. Create new checkpoint from current channel state
    self.checkpoint = create_checkpoint(
        self.checkpoint,
        self.channels if do_checkpoint else None,  # only snapshot channels if saving
        self.step,
        id=self.checkpoint["id"] if exiting else None,  # keep ID if exiting
        updated_channels=self.updated_channels,
    )

    # 5. Actually save to checkpointer
    if do_checkpoint:
        channel_versions = self.checkpoint["channel_versions"].copy()
        new_versions = get_new_channel_versions(
            self.checkpoint_previous_versions, channel_versions
        )
        self.checkpoint_previous_versions = channel_versions

        # Submit to background executor (non-blocking for "async" durability)
        self._put_checkpoint_fut = self.submit(
            self._checkpointer_put_after_previous,
            previous_future,          # wait for previous save to finish first
            self.checkpoint_config,
            copy_checkpoint(self.checkpoint),
            self.checkpoint_metadata,
            new_versions,
        )
        self.checkpoint_config = patch_configurable(
            self.checkpoint_config,
            {CONFIG_KEY_CHECKPOINT_ID: self.checkpoint["id"]},
        )

    # 6. Increment step counter
    if not exiting:
        self.step += 1
```

**Key design — durability modes:**
- `"async"`: `put` is submitted to a background thread — next step begins immediately
- `"sync"`: `put` is submitted but previous `put` is awaited first — in-order
- `"exit"`: channel snapshots are skipped during the loop; only persisted on exit

---

# Part 4: Complete Lifecycle Trace

Here is `graph.invoke({"user_input": "hello"}, config={"configurable": {"thread_id": "t1"}})` with checkpointing:

```
INVOCATION 1 — FIRST CALL
==========================

1. Pregel.stream() creates SyncPregelLoop
2. SyncPregelLoop.__enter__():
   ├─ checkpointer.get_tuple(config) → None (first call for "t1")
   ├─ Create empty_checkpoint() — v=4, id=uuid, channel_values={}, step=-2
   ├─ channels_from_checkpoint(specs, empty_checkpoint)
   │  └─ Each channel.from_checkpoint(MISSING) → default value
   ├─ _first() applies input:
   │  └─ Maps {"user_input": "hello"} → channel writes
   │  └─ _put_checkpoint({"source": "input", "step": -1})
   │     └─ checkpointer.put(config, checkpoint, metadata, versions)
   │     └─ Storage: t1 → "" → {checkpoint_id_0: (checkpoint_b, metadata_b, None)}
   └─ Updated checkpoint_id_saved = checkpoint_id_0

3. Pregel loop begins:
   ├─ tick() — prepares tasks, checks interrupts
   ├─ [nodes execute: sanitize_input, classify_intent, policy_node, ...]
   └─ after_tick():
      ├─ apply_writes() — merge node outputs into channels
      ├─ _put_checkpoint({"source": "loop", "step": 0})
      │  └─ checkpointer.put(config, checkpoint_1, ...)
      │  └─ Storage: t1 → "" → {checkpoint_id_1: (...)}
      └─ checkpoint_id_saved = checkpoint_id_1

4. Step 1: tick() → [nodes] → after_tick() → _put_checkpoint(step=1)
5. Step 2: tick() → [nodes] → after_tick() → _put_checkpoint(step=2)
   ...
6. tick() returns False (no more tasks, status="done")

7. SyncPregelLoop.__exit__() — final cleanup, suppress_interrupt saves exit checkpoint


INVOCATION 2 — RESUMING FROM THREAD "t1"
===========================================

1. Pregel.stream() creates SyncPregelLoop
2. SyncPregelLoop.__enter__():
   ├─ checkpointer.get_tuple(config) → CheckpointTuple with checkpoint_latest
   │  └─ thread_id="t1", takes max(checkpoint_id) → latest checkpoint
   │  └─ Loads channel values from blobs
   │  └─ Loads pending_writes (if any were interrupted)
   ├─ STEP = checkpoint_metadata["step"] + 1 (continues from where it left off)
   ├─ channels_from_checkpoint(specs, loaded_checkpoint)
   │  └─ Each channel restored from its saved value
   ├─ _first() applies new input OR resumes:
   │  └─ If input provided: creates "input" checkpoint, applies writes
   │  └─ If resuming (input=None or Command): applies pending writes
   └─ Loop continues from STEP

3. Loop runs as before, appending new checkpoints


THE DATA ON DISK (InMemorySaver):

storage = {
    "t1": {                            # thread_id
        "": {                          # checkpoint_ns (empty = root)
            "uuid-0": (                # checkpoint_id
                b'{"v":4,"id":"uuid-0",...}',   # checkpoint sans values
                b'{"source":"input","step":-1}', # metadata
                None,                  # parent_checkpoint_id
            ),
            "uuid-1": (
                b'{"v":4,"id":"uuid-1",...}',
                b'{"source":"loop","step":0}',
                "uuid-0",              # parent = previous checkpoint
            ),
            "uuid-2": (
                b'{"v":4,"id":"uuid-2",...}',
                b'{"source":"loop","step":1}',
                "uuid-1",
            ),
        }
    }
}

blobs = {
    ("t1", "", "user_input", "0000000000000000001.0"): ("json", b'"hello"'),
    ("t1", "", "messages", "0000000000000000001.0"): ("json", b'[...]'),
    ("t1", "", "intent", "0000000000000000001.0"): ("json", b'"policy"'),
    # ... new versions for each changed channel per checkpoint
}
```

---

# Part 5: Interview Talking Points

## "How does LangGraph persist conversation state?"

> "LangGraph uses a `Checkpointer` abstraction. Every graph invocation with the same `thread_id` loads the previous run's state from the checkpointer. Inside the graph, each Pregel super-step saves a checkpoint — a snapshot of all channel values with version numbers. Channels that didn't change reuse the previous version's blob, so storage is efficient. Writes are accumulated per-task and flushed to the checkpointer via `put_writes()`."

## "What happens on the second call with the same thread_id?"

> "The `SyncPregelLoop.__enter__` calls `checkpointer.get_tuple(config)`. Since `thread_id` matches, it fetches the latest checkpoint for that thread. The step counter resumes from where it left off. Channels are restored via `from_checkpoint()` — each channel deserializes its saved value. Then the new input is applied, creating an 'input' checkpoint as a child of the previous one, and the loop continues."

## "How does time-travel work?"

> "If you pass `checkpoint_id` in the config, `get_tuple()` loads that exact checkpoint instead of the latest. This creates a 'fork' — the new execution branches from the historical checkpoint. The original timeline's checkpoints are untouched. The fork's checkpoint metadata has `source='fork'` and the parent points to the replayed checkpoint."

## "What are the durability tradeoffs?"

> "`async` (default): saves are submitted to a background thread — next step starts immediately, but a crash might lose the last step's writes. `sync`: each save waits for the previous one to finish, guaranteeing order. `exit`: only persists on graph exit — fastest execution but no recovery from mid-graph crashes. For production conversation agents, `async` is the right balance."

## "How does InMemorySaver work?"

> "It uses three nested defaultdicts: `storage[thread_id][checkpoint_ns][checkpoint_id]` maps to serialized checkpoint data. Channel values are stored separately as `blobs[(thread_id, checkpoint_ns, channel, version)]` — this de-duplicates unchanged values across checkpoints. Pending writes are stored in `writes[(thread_id, checkpoint_ns, checkpoint_id)]`."
