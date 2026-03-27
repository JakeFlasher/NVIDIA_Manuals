---
title: "cutlass.pipeline"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html#module-cutlass.pipeline"
---

# [cutlass.pipeline](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#module-cutlass.pipeline)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#module-cutlass.pipeline "Permalink to this headline")

```
_`class`_`cutlass.pipeline.``Agent`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.Agent "Link to this definition")
```

Bases: `Enum`

Agent indicates what is participating in the pipeline synchronization.

```
`Thread`_`=` `1`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.Agent.Thread "Link to this definition")
```

```
`ThreadBlock`_`=` `2`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.Agent.ThreadBlock "Link to this definition")
```

```
`ThreadBlockCluster`_`=` `3`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.Agent.ThreadBlockCluster "Link to this definition")
```

```
_`class`_`cutlass.pipeline.``CooperativeGroup`(
```

Bases: `object`

CooperativeGroup contains size and alignment restrictions for an Agent.

```
`__init__`(
```

```
_`class`_`cutlass.pipeline.``PipelineOp`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOp "Link to this definition")
```

Bases: `Enum`

PipelineOp assigns an operation to an agent corresponding to a specific hardware feature.

```
`AsyncThread`_`=` `1`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOp.AsyncThread "Link to this definition")
```

```
`TCGen05Mma`_`=` `2`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOp.TCGen05Mma "Link to this definition")
```

```
`TmaLoad`_`=` `3`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOp.TmaLoad "Link to this definition")
```

```
`ClcLoad`_`=` `4`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOp.ClcLoad "Link to this definition")
```

```
`TmaStore`_`=` `5`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOp.TmaStore "Link to this definition")
```

```
`Composite`_`=` `6`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOp.Composite "Link to this definition")
```

```
`AsyncLoad`_`=` `7`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOp.AsyncLoad "Link to this definition")
```

```
_`class`_`cutlass.pipeline.``SyncObject`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject "Link to this definition")
```

Bases: `ABC`

Abstract base class for hardware synchronization primitives.

This class defines the interface for different types of hardware synchronization
mechanisms including shared memory barriers, named barriers, and fences.

```
_`abstract`_`arrive`() → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject.arrive "Link to this definition")
```

```
_`abstract`_`wait`() → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject.wait "Link to this definition")
```

```
_`abstract`_`arrive_and_wait`() → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject.arrive_and_wait "Link to this definition")
```

```
_`abstract`_`arrive_and_drop`() → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject.arrive_and_drop "Link to this definition")
```

```
_`abstract`_`get_barrier`() → `cutlass.cute.typing.Pointer` `|` `int` `|` `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject.get_barrier "Link to this definition")
```

```
_`abstract`_`max`() → `int` `|` `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject.max "Link to this definition")
```

```
`_abc_impl`_`=` `<_abc._abc_data` `object>`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject._abc_impl "Link to this definition")
```

```
_`class`_`cutlass.pipeline.``MbarrierArray`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.MbarrierArray "Link to this definition")
```

Bases: [`SyncObject`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")

MbarrierArray implements an abstraction for an array of smem barriers.

```
`__init__`(
```

```
`recast_to_new_op_type`(
```

Creates a copy of MbarrierArray with a different op_type without re-initializing barriers

```
`mbarrier_init`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.MbarrierArray.mbarrier_init "Link to this definition")
```

Initializes an array of mbarriers using warp 0.

```
`arrive`(
```

Select the arrive corresponding to this MbarrierArray’s PipelineOp.

**Parameters:**
: - **index** (_int_) – Index of the mbarrier in the array to arrive on
- **dst** (_int__|__None_) – Destination parameter for selective arrival, which can be either a mask or destination cta rank.
When None, both `TCGen05Mma` and `AsyncThread` will arrive on their local mbarrier.
- For `TCGen05Mma`, `dst` serves as a multicast mask (e.g., 0b1011 allows arrive signal to be multicast to CTAs
in the cluster with rank = 0, 1, and 3).
- For `AsyncThread`, `dst` serves as a destination cta rank (e.g., 3 means threads will arrive on
the mbarrier with rank = 3 in the cluster).
- **cta_group** (`cute.nvgpu.tcgen05.CtaGroup`, optional) – CTA group for `TCGen05Mma`, defaults to None for other op types

```
`arrive_mbarrier`(
```

```
`arrive_cp_async_mbarrier`(
```

```
`arrive_tcgen05mma`(
```

```
`arrive_and_expect_tx`(
```

```
`arrive_and_expect_tx_with_dst`(
```

```
`try_wait`(
```

```
`wait`(
```

```
`arrive_and_wait`(
```

```
`arrive_and_drop`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.MbarrierArray.arrive_and_drop "Link to this definition")
```

```
`get_barrier`(
```

```
`max`() → `int`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.MbarrierArray.max "Link to this definition")
```

```
`_abc_impl`_`=` `<_abc._abc_data` `object>`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.MbarrierArray._abc_impl "Link to this definition")
```

```
_`class`_`cutlass.pipeline.``NamedBarrier`(_`barrier_id``:` `int`_, _`num_threads``:` `int`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.NamedBarrier "Link to this definition")
```

Bases: [`SyncObject`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")

NamedBarrier is an abstraction for named barriers managed by hardware.
There are 16 named barriers available, with barrier_ids 0-15.

See the [PTX documentation](https://https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar).

```
`barrier_id`_`:` `int`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.NamedBarrier.barrier_id "Link to this definition")
```

```
`num_threads`_`:` `int`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.NamedBarrier.num_threads "Link to this definition")
```

```
`arrive`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.NamedBarrier.arrive "Link to this definition")
```

The aligned flavor of arrive is used when all threads in the CTA will execute the
same instruction. See PTX documentation.

```
`arrive_unaligned`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.NamedBarrier.arrive_unaligned "Link to this definition")
```

The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.

```
`wait`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.NamedBarrier.wait "Link to this definition")
```

NamedBarriers do not have a standalone wait like mbarriers, only an arrive_and_wait.
If synchronizing two warps in a producer/consumer pairing, the arrive count would be
32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer
or consumer are counted for mbarriers, while all threads participating in the sync
are counted for NamedBarriers.

```
`wait_unaligned`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.NamedBarrier.wait_unaligned "Link to this definition")
```

```
`arrive_and_wait`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.NamedBarrier.arrive_and_wait "Link to this definition")
```

```
`arrive_and_drop`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.NamedBarrier.arrive_and_drop "Link to this definition")
```

```
`sync`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.NamedBarrier.sync "Link to this definition")
```

```
`get_barrier`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `int`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.NamedBarrier.get_barrier "Link to this definition")
```

```
`max`() → `int`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.NamedBarrier.max "Link to this definition")
```

```
`__init__`(_`barrier_id``:` `int`_, _`num_threads``:` `int`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.NamedBarrier.__init__ "Link to this definition")
```

```
`_abc_impl`_`=` `<_abc._abc_data` `object>`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.NamedBarrier._abc_impl "Link to this definition")
```

```
_`class`_`cutlass.pipeline.``PipelineOrder`(
```

Bases: `object`

PipelineOrder is used for managing ordered pipeline execution with multiple groups.

This class implements a pipeline ordering mechanism where work is divided into groups
and stages, allowing for controlled progression through pipeline stages with proper
synchronization between different groups.

The pipeline ordering works as follows:
- The pipeline is divided into ‘length’ number of groups
- Each group has ‘depth’ number of stages
- Groups execute in a specific order with synchronization barriers
- Each group waits for the previous group to complete before proceeding

**Example:**

```python
# Create pipeline order with 3 groups, each with 2 stages
pipeline_order = PipelineOrder.create(
    barrier_storage=smem_ptr,      # shared memory pointer for barriers
    depth=2,                       # 2 stages per group
    length=3,                      # 3 groups total
    group_id=0,                    # current group ID (0, 1, or 2)
    producer_group=producer_warp   # cooperative group for producers
)

# In the pipeline loop
for stage in range(num_stages):
    pipeline_order.wait()          # Wait for previous group to complete
    # Process current stage
    pipeline_order.arrive()        # Signal completion to next group
```

```
`sync_object_full`_`:` [`SyncObject`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOrder.sync_object_full "Link to this definition")
```

```
`depth`_`:` `int`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOrder.depth "Link to this definition")
```

```
`length`_`:` `int`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOrder.length "Link to this definition")
```

```
`group_id`_`:` `int`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOrder.group_id "Link to this definition")
```

```
`state`_`:` [`PipelineState`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOrder.state "Link to this definition")
```

```
_`static`_`create`(
```

```
`get_barrier_for_current_stage_idx`(_`group_id`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOrder.get_barrier_for_current_stage_idx "Link to this definition")
```

```
`arrive`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOrder.arrive "Link to this definition")
```

```
`wait`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineOrder.wait "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.pipeline.``TmaStoreFence`(_`num_stages``:` `int` `=` `0`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.TmaStoreFence "Link to this definition")
```

Bases: [`SyncObject`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")

TmaStoreFence is used for a multi-stage epilogue buffer.

```
`__init__`(_`num_stages``:` `int` `=` `0`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.TmaStoreFence.__init__ "Link to this definition")
```

```
`arrive`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.TmaStoreFence.arrive "Link to this definition")
```

```
`wait`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.TmaStoreFence.wait "Link to this definition")
```

```
`arrive_and_wait`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.TmaStoreFence.arrive_and_wait "Link to this definition")
```

```
`arrive_and_drop`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.TmaStoreFence.arrive_and_drop "Link to this definition")
```

```
`get_barrier`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.TmaStoreFence.get_barrier "Link to this definition")
```

```
`max`() → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.TmaStoreFence.max "Link to this definition")
```

```
`tail`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.TmaStoreFence.tail "Link to this definition")
```

```
`_abc_impl`_`=` `<_abc._abc_data` `object>`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.TmaStoreFence._abc_impl "Link to this definition")
```

```
_`class`_`cutlass.pipeline.``PipelineUserType`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineUserType "Link to this definition")
```

Bases: `Enum`

An enumeration.

```
`Producer`_`=` `1`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineUserType.Producer "Link to this definition")
```

```
`Consumer`_`=` `2`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineUserType.Consumer "Link to this definition")
```

```
`ProducerConsumer`_`=` `3`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineUserType.ProducerConsumer "Link to this definition")
```

```
_`class`_`cutlass.pipeline.``PipelineState`(_`stages``:` `int`_, _`count`_, _`index`_, _`phase`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState "Link to this definition")
```

Bases: `object`

Pipeline state contains an index and phase bit corresponding to the current position in the circular buffer.

```
`__init__`(_`stages``:` `int`_, _`count`_, _`index`_, _`phase`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState.__init__ "Link to this definition")
```

```
`clone`() → [`PipelineState`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState.clone "Link to this definition")
```

```
_`property`_`index`_`:` `cutlass.cutlass_dsl.Int32`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState.index "Link to this definition")
```

```
_`property`_`count`_`:` `cutlass.cutlass_dsl.Int32`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState.count "Link to this definition")
```

```
_`property`_`stages`_`:` `int`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState.stages "Link to this definition")
```

```
_`property`_`phase`_`:` `cutlass.cutlass_dsl.Int32`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState.phase "Link to this definition")
```

```
`reset_count`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState.reset_count "Link to this definition")
```

```
`advance`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState.advance "Link to this definition")
```

```
`reverse`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState.reverse "Link to this definition")
```

```
_`class`_`cutlass.pipeline.``PipelineAsync`(
```

Bases: `object`

PipelineAsync is a generic pipeline class where both the producer and consumer are
AsyncThreads. It also serves as a base class for specialized pipeline classes.

This class implements a producer-consumer pipeline pattern where both sides operate
asynchronously. The pipeline maintains synchronization state using barrier objects
to coordinate between producer and consumer threads.

The pipeline state transitions of one pipeline entry(mbarrier) can be represented as:

| Barrier | State | p.acquire | p.commit | c.wait | c.release |
| --- | --- | --- | --- | --- | --- |
| empty_bar | empty | <Return> | n/a | n/a | - |
| empty_bar | wait | <Block> | n/a | n/a | -> empty |
| full_bar | wait | n/a | -> full | <Block > | n/a |
| full_bar | full | n/a | - | <Return> | n/a |

Where:

- p: producer
- c: consumer
- <Block>: This action is blocked until transition to a state allow it to proceed by other side
- e.g. `p.acquire()` is blocked until `empty_bar` transition to `empty` state by `c.release()`

```text
Array of mbarriers as circular buffer:

     Advance Direction
   <-------------------

    Producer   Consumer
        |         ^
        V         |
   +-----------------+
 --|X|X|W|D|D|D|D|R|X|<-.
/  +-----------------+   \
|                        |
`------------------------'
```

Where:

- X: Empty buffer (initial state)
- W: Producer writing (producer is waiting for buffer to be empty)
- D: Data ready (producer has written data to buffer)
- R: Consumer reading (consumer is consuming data from buffer)

**Example:**

```python
# Create pipeline with 5 stages
pipeline = PipelineAsync.create(
    num_stages=5,                   # number of pipeline stages
    producer_group=producer_warp,
    consumer_group=consumer_warp
    barrier_storage=smem_ptr,       # smem pointer for array of mbarriers in shared memory
)

producer, consumer = pipeline.make_participants()
# Producer side
for i in range(num_iterations):
    handle = producer.acquire_and_advance()  # Wait for buffer to be empty & Move index to next stage
    # Write data to pipeline buffer
    handle.commit()   # Signal buffer is full

# Consumer side
for i in range(num_iterations):
    handle = consumer.wait_and_advance()     # Wait for buffer to be full & Move index to next stage
    # Read data from pipeline buffer
    handle.release()  # Signal buffer is empty
```

```
`sync_object_full`_`:` [`SyncObject`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync.sync_object_full "Link to this definition")
```

```
`sync_object_empty`_`:` [`SyncObject`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync.sync_object_empty "Link to this definition")
```

```
`num_stages`_`:` `int`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync.num_stages "Link to this definition")
```

```
`producer_mask`_`:` `cutlass.cutlass_dsl.Int32` `|` `None`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync.producer_mask "Link to this definition")
```

```
`consumer_mask`_`:` `cutlass.cutlass_dsl.Int32` `|` `None`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync.consumer_mask "Link to this definition")
```

```
_`static`_`_make_sync_object`(
```

Returns a SyncObject corresponding to an agent’s PipelineOp.

```
_`static`_`create`(
```

Creates and initializes a new PipelineAsync instance.

This helper function computes necessary attributes and returns an instance of PipelineAsync
with the specified configuration for producer and consumer synchronization.

**Parameters:**
: - **barrier_storage** (_cute.Pointer_) – Pointer to the shared memory address for this pipeline’s mbarriers
- **num_stages** (_int_) – Number of buffer stages for this pipeline
- **producer_group** ([_CooperativeGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the producer agent
- **consumer_group** ([_CooperativeGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the consumer agent
- **producer_mask** (_Int32__,__optional_) – Mask for signaling arrives for the producer agent
- **consumer_mask** (_Int32__,__optional_) – Mask for signaling arrives for the consumer agent

**Raises:**
: **ValueError** – If barrier_storage is not a cute.Pointer instance

**Returns:**
: A new `PipelineAsync` instance

**Return type:**
: [PipelineAsync](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync "cutlass.pipeline.PipelineAsync")

```
`producer_acquire`(
```

```
`producer_try_acquire`(
```

```
`producer_commit`(
```

```
`consumer_wait`(
```

```
`consumer_try_wait`(
```

```
`consumer_release`(
```

```
`producer_get_barrier`(
```

```
`producer_tail`(
```

Make sure the last used buffer empty signal is visible to producer.
Producer tail is usually executed by producer before exit, to avoid dangling
mbarrier arrive signals after kernel exit.

**Parameters:**
: **state** ([_PipelineState_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState "cutlass.pipeline.PipelineState")) – The pipeline state that points to next useful buffer

```
`make_producer`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync.make_producer "Link to this definition")
```

```
`make_consumer`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync.make_consumer "Link to this definition")
```

```
`make_participants`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync.make_participants "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.pipeline.``PipelineCpAsync`(
```

Bases: [`PipelineAsync`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

PipelineCpAsync is used for CpAsync producers and AsyncThread consumers

```
_`static`_`create`(
```

Helper function that computes necessary attributes and returns a `PipelineCpAsync` instance.

**Parameters:**
: - **barrier_storage** (_cute.Pointer_) – Pointer to the shared memory address for this pipeline’s mbarriers
- **num_stages** (_Int32_) – Number of buffer stages for this pipeline
- **producer_group** ([_CooperativeGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the producer agent
- **consumer_group** ([_CooperativeGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the consumer agent
- **producer_mask** (_Int32__,__optional_) – Mask for signaling arrives for the producer agent, defaults to None
- **consumer_mask** (_Int32__,__optional_) – Mask for signaling arrives for the consumer agent, defaults to None

**Returns:**
: A new `PipelineCpAsync` instance configured with the provided parameters

**Return type:**
: [PipelineCpAsync](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineCpAsync "cutlass.pipeline.PipelineCpAsync")

```
`__init__`(
```

```
_`class`_`cutlass.pipeline.``PipelineTmaAsync`(
```

Bases: [`PipelineAsync`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

PipelineTmaAsync is used for TMA producers and AsyncThread consumers (e.g. Hopper mainloops).

```
`is_signalling_thread`_`:` `cutlass.cutlass_dsl.Boolean`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineTmaAsync.is_signalling_thread "Link to this definition")
```

```
_`static`_`init_empty_barrier_arrive_signal`(
```

Initialize the empty barrier arrive signal.

This function determines which threads should signal empty barrier arrives based on the cluster layout
and multicast modes. It returns the destination CTA rank and whether the current thread should signal.

**Parameters:**
: - **cta_layout_vmnk** (_cute.Layout_) – Layout describing the cluster shape and CTA arrangement
- **tidx** (_Int32_) – Thread index within the warp
- **mcast_mode_mn** (_tuple__[__int__,__int__]_) – Tuple specifying multicast modes for m and n dimensions (each 0 or 1), defaults to (1,1)

**Raises:**
: **AssertionError** – If both multicast modes are disabled (0,0)

**Returns:**
: Tuple containing destination CTA rank and boolean indicating if current thread signals

**Return type:**
: tuple[Int32, Boolean]

```
_`static`_`create`(
```

Create a new `PipelineTmaAsync` instance.

**Parameters:**
: - **num_stages** (_int_) – Number of buffer stages for this pipeline
- **producer_group** ([_CooperativeGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the producer agent
- **consumer_group** ([_CooperativeGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the consumer agent
- **tx_count** (_int_) – Number of bytes expected to be written to the transaction barrier for one stage
- **barrier_storage** (_cute.Pointer__,__optional_) – Pointer to the shared memory address for this pipeline’s mbarriers, defaults to None
- **cta_layout_vmnk** (_cute.Layout__,__optional_) – Layout of the cluster shape, defaults to None
- **tidx** (_Int32__,__optional_) – Thread index to consumer async threads, defaults to None
- **mcast_mode_mn** (_tuple__[__int__,__int__]__,__optional_) – Tuple specifying multicast modes for m and n dimensions (each 0 or 1), defaults to (1,1)

**Raises:**
: **ValueError** – If barrier_storage is not a cute.Pointer instance

**Returns:**
: New `PipelineTmaAsync` instance

**Return type:**
: [PipelineTmaAsync](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineTmaAsync "cutlass.pipeline.PipelineTmaAsync")

```
`producer_acquire`(
```

TMA producer commit conditionally waits on buffer empty and sets the transaction barrier.

```
`producer_commit`(
```

TMA producer commit is a noop since TMA instruction itself updates the transaction count.

```
`consumer_release`(
```

TMA consumer release conditionally signals the empty buffer to the producer.

```
`__init__`(
```

```
_`class`_`cutlass.pipeline.``PipelineTmaUmma`(
```

Bases: [`PipelineAsync`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

PipelineTmaUmma is used for TMA producers and UMMA consumers (e.g. Blackwell mainloops).

```
`is_leader_cta`_`:` `bool`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineTmaUmma.is_leader_cta "Link to this definition")
```

```
`cta_group`_`:` [`CtaGroup`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineTmaUmma.cta_group "Link to this definition")
```

```
`_make_sync_object`(
```

Returns a SyncObject corresponding to an agent’s PipelineOp.

```
`_compute_mcast_arrival_mask`(
```

Computes a mask for signaling arrivals to multicasting threadblocks.

```
`_compute_is_leader_cta`(
```

Computes leader threadblocks for 2CTA kernels. For 1CTA, all threadblocks are leaders.

```
`create`(
```

Creates and initializes a new PipelineTmaUmma instance.

**Parameters:**
: - **num_stages** (_int_) – Number of buffer stages for this pipeline
- **producer_group** ([_CooperativeGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – CooperativeGroup for the producer agent
- **consumer_group** ([_CooperativeGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – CooperativeGroup for the consumer agent
- **tx_count** (_int_) – Number of bytes expected to be written to the transaction barrier for one stage
- **barrier_storage** (_cute.Pointer__,__optional_) – Pointer to the shared memory address for this pipeline’s mbarriers
- **cta_layout_vmnk** (_cute.Layout__,__optional_) – Layout of the cluster shape
- **mcast_mode_mn** (_tuple__[__int__,__int__]__,__optional_) – Tuple specifying multicast modes for m and n dimensions (each 0 or 1)

**Raises:**
: **ValueError** – If barrier_storage is not a cute.Pointer instance

**Returns:**
: A new PipelineTmaUmma instance configured with the provided parameters

**Return type:**
: [PipelineTmaUmma](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineTmaUmma "cutlass.pipeline.PipelineTmaUmma")

```
`consumer_release`(
```

UMMA consumer release buffer empty, cta_group needs to be provided.

```
`producer_acquire`(
```

TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.

```
`producer_commit`(
```

TMA producer commit is a noop since TMA instruction itself updates the transaction count.

```
`__init__`(
```

```
_`class`_`cutlass.pipeline.``PipelineAsyncUmma`(
```

Bases: [`PipelineAsync`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

PipelineAsyncUmma is used for AsyncThread producers and UMMA consumers (e.g. Blackwell input fusion pipelines).

```
`cta_group`_`:` [`CtaGroup`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsyncUmma.cta_group "Link to this definition")
```

```
`_compute_leading_cta_rank`(
```

Computes the leading CTA rank.

```
`_compute_is_leader_cta`(
```

Computes leader threadblocks for 2CTA kernels. For 1CTA, all threadblocks are leaders.

```
`_compute_peer_cta_mask`(
```

Computes a mask for signaling arrivals to multicasting threadblocks.

```
`create`(
```

Creates and initializes a new PipelineAsyncUmma instance.

**Parameters:**
: - **num_stages** (_int_) – Number of buffer stages for this pipeline
- **producer_group** ([_CooperativeGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – CooperativeGroup for the producer agent
- **consumer_group** ([_CooperativeGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – CooperativeGroup for the consumer agent
- **barrier_storage** (_cute.Pointer__,__optional_) – Pointer to the shared memory address for this pipeline’s mbarriers
- **cta_layout_vmnk** (_cute.Layout__,__optional_) – Layout of the cluster shape

**Raises:**
: **ValueError** – If barrier_storage is not a cute.Pointer instance

**Returns:**
: A new PipelineAsyncUmma instance configured with the provided parameters

**Return type:**
: [PipelineAsyncUmma](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsyncUmma "cutlass.pipeline.PipelineAsyncUmma")

```
`consumer_release`(
```

UMMA consumer release buffer empty, cta_group needs to be provided.

```
`__init__`(
```

```
_`class`_`cutlass.pipeline.``PipelineUmmaAsync`(
```

Bases: [`PipelineAsync`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

PipelineUmmaAsync is used for UMMA producers and AsyncThread consumers (e.g. Blackwell accumulator pipelines).

```
`cta_group`_`:` [`CtaGroup`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineUmmaAsync.cta_group "Link to this definition")
```

```
`_compute_tmem_sync_mask`(
```

Computes a mask to signal completion of tmem buffers for 2CTA kernels.

```
`_compute_peer_cta_rank`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineUmmaAsync._compute_peer_cta_rank "Link to this definition")
```

Computes a mask to signal release of tmem buffers for 2CTA kernels.

```
`create`(
```

Creates an instance of PipelineUmmaAsync with computed attributes.

**Parameters:**
: - **num_stages** (_int_) – Number of buffer stages for this pipeline
- **producer_group** ([_CooperativeGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the producer agent
- **consumer_group** ([_CooperativeGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the consumer agent
- **barrier_storage** (_cute.Pointer__,__optional_) – Pointer to the shared memory address for this pipeline’s mbarriers
- **cta_layout_vmnk** (_cute.Layout__,__optional_) – Layout of the cluster shape

**Raises:**
: **ValueError** – If barrier_storage is not a cute.Pointer instance

**Returns:**
: New instance of `PipelineUmmaAsync`

**Return type:**
: [PipelineUmmaAsync](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineUmmaAsync "cutlass.pipeline.PipelineUmmaAsync")

```
`producer_commit`(
```

UMMA producer commit buffer full, cta_group needs to be provided.

```
`__init__`(
```

```
_`class`_`cutlass.pipeline.``PipelineClcFetchAsync`(
```

Bases: `object`

PipelineClcFetchAsync implements a producer-consumer pipeline for Cluster Launch
Control based dynamic scheduling. Both producer and consumer operate asynchronously
using barrier synchronization to coordinate across pipeline stages and cluster CTAs.

- Producer: waits for empty buffer, signals full barrier with transection bytes
across all CTAs in cluster, hardware autosignals each CTA’s mbarrier when
transaction bytes are written, then the satte advance to next buffer slot.
- Consumer: waits for full barrier, then load respinse from local SMEM, then
sigals CTA 0’s empty barrier to allow buffer reuse.

```
`sync_object_full`_`:` [`SyncObject`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineClcFetchAsync.sync_object_full "Link to this definition")
```

```
`sync_object_empty`_`:` [`SyncObject`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineClcFetchAsync.sync_object_empty "Link to this definition")
```

```
`num_stages`_`:` `int`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineClcFetchAsync.num_stages "Link to this definition")
```

```
`producer_mask`_`:` `cutlass.cutlass_dsl.Int32` `|` `None`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineClcFetchAsync.producer_mask "Link to this definition")
```

```
`consumer_mask`_`:` `cutlass.cutlass_dsl.Int32` `|` `None`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineClcFetchAsync.consumer_mask "Link to this definition")
```

```
`is_signalling_thread`_`:` `cutlass.cutlass_dsl.Boolean`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineClcFetchAsync.is_signalling_thread "Link to this definition")
```

```
_`static`_`_init_full_barrier_arrive_signal`(
```

Computes producer barrier signaling parameters, returns destination CTA rank
(0 to cluster_size-1) based on thread ID, and a boolean flag indicating if
this thread participates in signaling.

**Parameters:**
: - **cta_layout_vmnk** – Cluster layout defining CTA count
- **tidx** – Thread ID within the CTA

```
_`static`_`create`(
```

This helper function computes any necessary attributes and returns an instance of PipelineClcFetchAsync.
:param barrier_storage: Pointer to the shared memory address for this pipeline’s mbarriers
:type barrier_storage: cute.Pointer
:param num_stages: Number of buffer stages for this pipeline
:type num_stages: int
:param producer_group: *CooperativeGroup* for the producer agent
:type producer_group: CooperativeGroup
:param consumer_group: *CooperativeGroup* for the consumer agent
:type consumer_group: CooperativeGroup
:param tx_count: Number of bytes expected to be written to the transaction barrier for one stage
:type tx_count: int
:param producer_mask: Mask for signaling arrives for the producer agent, defaults to `None`
:type producer_mask: Int32, optional
:param consumer_mask: Mask for signaling arrives for the consumer agent, defaults to `None`
:type consumer_mask: Int32, optional

```
`producer_acquire`(
```

Producer acquire waits for empty buffer and sets transaction expectation on full barrier.

**Parameters:**
: - **state** – Pipeline state pointing to the current buffer stage
- **try_acquire_token** – Optional token to skip the empty barrier wait

```
`consumer_wait`(
```

Consumer waits for full barrier to be signaled by hardware multicast.

**Parameters:**
: - **state** – Pipeline state pointing to the current buffer stage
- **try_wait_token** – Optional token to skip the full barrier wait

```
`consumer_release`(
```

```
`producer_get_barrier`(
```

```
`producer_tail`(
```

Ensures all in-flight buffers are released before producer exits.

**Parameters:**
: - **state** – Pipeline state with current position in the buffer
- **try_acquire_token** – Optional token to skip the empty barrier waits

```
`__init__`(
```

```
_`class`_`cutlass.pipeline.``PipelineTmaMultiConsumersAsync`(
```

Bases: [`PipelineAsync`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

PipelineTmaMultiConsumersAsync is used for TMA producers and UMMA+Async consumers.

```
`is_leader_cta`_`:` `bool`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineTmaMultiConsumersAsync.is_leader_cta "Link to this definition")
```

```
`sync_object_empty_umma`_`:` [`SyncObject`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineTmaMultiConsumersAsync.sync_object_empty_umma "Link to this definition")
```

```
`sync_object_empty_async`_`:` [`SyncObject`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineTmaMultiConsumersAsync.sync_object_empty_async "Link to this definition")
```

```
`cta_group`_`:` [`CtaGroup`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineTmaMultiConsumersAsync.cta_group "Link to this definition")
```

```
_`static`_`create`(
```

This helper function computes any necessary attributes and returns an instance of PipelineTmaMultiConsumersAsync.
:param barrier_storage: Pointer to the smem address for this pipeline’s mbarriers
:type barrier_storage: cute.Pointer
:param num_stages: Number of buffer stages for this pipeline
:type num_stages: Int32
:param producer_group: *CooperativeGroup* for the producer agent
:type producer_group: CooperativeGroup
:param consumer_group_umma: *CooperativeGroup* for the UMMA consumer agent
:type consumer_group_umma: CooperativeGroup
:param consumer_group_async: *CooperativeGroup* for the AsyncThread consumer agent
:type consumer_group_async: CooperativeGroup
:param tx_count: Number of bytes expected to be written to the transaction barrier for one stage
:type tx_count: int
:param cta_layout_vmnk: Layout of the cluster shape
:type cta_layout_vmnk: cute.Layout | None

```
`producer_acquire`(
```

TMA producer acquire waits on buffer empty and sets the transaction barrier for leader threadblocks.

```
`producer_commit`(
```

TMA producer commit is a noop since TMA instruction itself updates the transaction count.

```
`consumer_release`(
```

```
`__init__`(
```

```
_`class`_`cutlass.pipeline.``PipelineTmaStore`(
```

Bases: [`PipelineAsync`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

PipelineTmaStore is used for synchronizing TMA stores in the epilogue. It does not use mbarriers.

```
_`static`_`create`(
```

This helper function computes any necessary attributes and returns an instance of `PipelineTmaStore`.

**Parameters:**
: - **num_stages** (_int_) – Number of buffer stages for this pipeline
- **producer_group** ([_CooperativeGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the producer agent

**Returns:**
: A new `PipelineTmaStore` instance

**Return type:**
: [PipelineTmaStore](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineTmaStore "cutlass.pipeline.PipelineTmaStore")

```
`producer_acquire`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineTmaStore.producer_acquire "Link to this definition")
```

```
`producer_commit`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineTmaStore.producer_commit "Link to this definition")
```

```
`consumer_wait`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineTmaStore.consumer_wait "Link to this definition")
```

```
`consumer_release`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineTmaStore.consumer_release "Link to this definition")
```

```
`producer_tail`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineTmaStore.producer_tail "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.pipeline.``PipelineProducer`(
```

Bases: `object`

A class representing a producer in an asynchronous pipeline.

This class manages the producer side of an asynchronous pipeline, handling
synchronization and state management for producing data. It provides methods for
acquiring, committing, and advancing through pipeline stages.

**Variables:**
: - **__pipeline** – The asynchronous pipeline this producer belongs to
- **__state** – The current state of the producer in the pipeline
- **__group** – The cooperative group this producer operates in

**Examples:**

```python
pipeline = PipelineAsync.create(...)
producer, consumer = pipeline.make_participants()
for i in range(iterations):
    # Try to acquire the current buffer without blocking
    try_acquire_token = producer.try_acquire()

    # Do something else independently
    ...

    # Wait for current buffer to be empty & Move index to next stage
    # If try_acquire_token is True, return immediately
    # If try_acquire_token is False, block until buffer is empty
    handle = producer.acquire_and_advance(try_acquire_token)

    # Produce data
    handle.commit()
```

```
_`class`_`ImmutableResourceHandle`(
```

Bases: `ImmutableResourceHandle`

```
_`property`_`barrier`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle.barrier "Link to this definition")
```

Get the barrier pointer for the current pipeline stage.

**Returns:**
: Pointer to the barrier for the current stage

**Return type:**
: cute.Pointer

```
`commit`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle.commit "Link to this definition")
```

Signal that data production is complete for the current stage.

This allows consumers to start processing the data.

```
`__init__`(
```

```
`__init__`(
```

Initialize a new Producer instance.

**Parameters:**
: - **pipeline** ([_PipelineAsync_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync "cutlass.pipeline.PipelineAsync")) – The pipeline this producer belongs to
- **state** ([_PipelineState_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState "cutlass.pipeline.PipelineState")) – Initial pipeline state
- **group** ([_CooperativeGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – The cooperative group for synchronization

```
`__pipeline`_`:` [`PipelineAsync`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineProducer.__pipeline "Link to this definition")
```

```
`__state`_`:` [`PipelineState`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineProducer.__state "Link to this definition")
```

```
`__group`_`:` [`CooperativeGroup`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineProducer.__group "Link to this definition")
```

```
`clone`()[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineProducer.clone "Link to this definition")
```

Create a new Producer instance with the same state.

```
`reset`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineProducer.reset "Link to this definition")
```

Reset the count of how many handles this producer has committed.

```
`acquire`(
```

Wait for the current buffer to be empty before producing data.
This is a blocking operation.

**Parameters:**
: **try_acquire_token** (_Optional__[__Boolean__]_) – Optional token to try to acquire the buffer

**Returns:**
: A handle to the producer for committing the data

**Return type:**
: [ImmutableResourceHandle](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.PipelineProducer.ImmutableResourceHandle")

```
`advance`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineProducer.advance "Link to this definition")
```

Move to the next pipeline stage.

```
`acquire_and_advance`(
```

Acquire the current buffer and advance to the next pipeline stage.

This method combines the acquire() and advance() operations into a single call.
It first waits for the current buffer to be empty before producing data,
then advances the pipeline to the next stage.

**Parameters:**
: **try_acquire_token** (_Optional__[__Boolean__]_) – Token indicating whether to try non-blocking acquire.
If True, returns immediately without waiting. If False or None, blocks
until buffer is empty.

**Returns:**
: A handle to the producer that can be used to commit data to the
acquired buffer stage

**Return type:**
: [ImmutableResourceHandle](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.PipelineProducer.ImmutableResourceHandle")

```
`try_acquire`(
```

Attempt to acquire the current buffer without blocking.

This method tries to acquire the current buffer stage for producing data
without waiting. It can be used to check buffer availability before
committing to a blocking acquire operation.

**Returns:**
: A boolean token indicating whether the buffer was successfully acquired

**Return type:**
: Boolean

```
`commit`(
```

Signal that data production is complete for the current stage.

This allows consumers to start processing the data.

**Parameters:**
: **handle** (_Optional__[_[_ImmutableResourceHandle_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.PipelineProducer.ImmutableResourceHandle")_]_) – Optional handle to commit, defaults to None

**Raises:**
: **AssertionError** – If provided handle does not belong to this producer

```
`tail`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineProducer.tail "Link to this definition")
```

Ensure all used buffers are properly synchronized before producer exit.

This should be called before the producer finishes to avoid dangling signals.

```
_`class`_`cutlass.pipeline.``PipelineConsumer`(
```

Bases: `object`

A class representing a consumer in an asynchronous pipeline.

The Consumer class manages the consumer side of an asynchronous pipeline, handling
synchronization and state management for consuming data. It provides methods for
waiting, releasing, and advancing through pipeline stages.

**Variables:**
: - **__pipeline** – The asynchronous pipeline this consumer belongs to
- **__state** – The current state of the consumer in the pipeline
- **__group** – The cooperative group this consumer operates in

**Examples:**

```python
pipeline = PipelineAsync.create(...)
producer, consumer = pipeline.make_participants()
for i in range(iterations):
    # Try to wait for buffer to be full
    try_wait_token = consumer.try_wait()

    # Do something else independently
    ...

    # Wait for buffer to be full & Move index to next stage
    # If try_wait_token is True, return immediately
    # If try_wait_token is False, block until buffer is full
    handle = consumer.wait_and_advance(try_wait_token)

    # Consume data
    handle.release(  )  # Signal buffer is empty

    # Alternative way to do this is:
    # handle.release()  # Signal buffer is empty
```

```
_`class`_`ImmutableResourceHandle`(
```

Bases: `ImmutableResourceHandle`

```
`release`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle.release "Link to this definition")
```

Signal that data production is complete for the current stage.
This allows consumers to start processing the data.

```
`__init__`(
```

```
`__init__`(
```

Initialize a new Consumer instance.

**Parameters:**
: - **pipeline** ([_PipelineAsync_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync "cutlass.pipeline.PipelineAsync")) – The pipeline this consumer belongs to
- **state** ([_PipelineState_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState "cutlass.pipeline.PipelineState")) – Initial pipeline state
- **group** ([_CooperativeGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – The cooperative group for synchronization

```
`__pipeline`_`:` [`PipelineAsync`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineConsumer.__pipeline "Link to this definition")
```

```
`__group`_`:` [`CooperativeGroup`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineConsumer.__group "Link to this definition")
```

```
`__state`_`:` [`PipelineState`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineConsumer.__state "Link to this definition")
```

```
`clone`()[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineConsumer.clone "Link to this definition")
```

Create a new Consumer instance with the same state.

```
`reset`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineConsumer.reset "Link to this definition")
```

Reset the count of how many handles this consumer has consumed.

```
`wait`(
```

Wait for data to be ready in the current buffer. This is a blocking operation
that will not return until data is available.

**Parameters:**
: **try_wait_token** (_Optional__[__Boolean__]_) – Token used to attempt a non-blocking wait for the buffer.
If provided and True, returns immediately if buffer is not ready.

**Returns:**
: An immutable handle to the consumer that can be used to release the buffer
once data consumption is complete

**Return type:**
: [ImmutableResourceHandle](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle")

```
`advance`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineConsumer.advance "Link to this definition")
```

Advance the consumer to the next pipeline stage.

This updates the internal state to point to the next buffer in the pipeline.
Should be called after consuming data from the current buffer.

```
`wait_and_advance`(
```

Atomically wait for data and advance to next pipeline stage.

This is a convenience method that combines wait() and advance() into a single
atomic operation. It will block until data is available in the current buffer,
then automatically advance to the next stage.

**Parameters:**
: **try_wait_token** (_Optional__[__Boolean__]_) – Token used to attempt a non-blocking wait for the buffer.
If provided and True, returns immediately if buffer is not ready.

**Returns:**
: An immutable handle to the consumer that can be used to release the buffer
once data consumption is complete

**Return type:**
: [ImmutableResourceHandle](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle")

```
`try_wait`(
```

Non-blocking check if data is ready in the current buffer.

This method provides a way to test if data is available without blocking.
Unlike wait(), this will return immediately regardless of buffer state.

**Returns:**
: True if data is ready to be consumed, False if the buffer is not yet ready

**Return type:**
: Boolean

```
`release`(
```

Signal that data consumption is complete for the current stage.
This allows producers to start producing new data.

```
`cutlass.pipeline.``make_pipeline_state`(
```

Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.

```
`cutlass.pipeline.``pipeline_init_arrive`(
```

Fences the mbarrier_init and sends an arrive if using clusters.

```
`cutlass.pipeline.``pipeline_init_wait`(
```

Syncs the threadblock or cluster

```
`cutlass.pipeline.``agent_sync`(
```

Syncs all threads within an agent.

```
`cutlass.pipeline.``arrive`(_`barrier_id``:` `int`_, _`num_threads``:` `int`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.arrive "Link to this definition")
```

The aligned flavor of arrive is used when all threads in the CTA will execute the
same instruction. See PTX documentation.

```
`cutlass.pipeline.``arrive_unaligned`(
```

The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.

```
`cutlass.pipeline.``wait`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.wait "Link to this definition")
```

NamedBarriers do not have a standalone wait like mbarriers, only an arrive_and_wait.
If synchronizing two warps in a producer/consumer pairing, the arrive count would be
32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer
or consumer are counted for mbarriers, while all threads participating in the sync
are counted for NamedBarriers.

```
`cutlass.pipeline.``wait_unaligned`(
```

```
`cutlass.pipeline.``arrive_and_wait`(
```

```
`cutlass.pipeline.``sync`(_`barrier_id``:` `int` `=` `0`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.pipeline.sync "Link to this definition")
```
