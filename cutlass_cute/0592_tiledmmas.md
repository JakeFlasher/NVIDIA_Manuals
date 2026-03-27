---
title: "TiledMMAs"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html#tiledmmas"
---

## [TiledMMAs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#tiledmmas)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#tiledmmas "Permalink to this headline")

We can make more complex patterns by combining and interleaving multiple atoms.

Let’s start with `SM70_8x8x4_F32F16F16F32_NT`.

```cpp
MMA_Atom mma = MMA_Atom<SM70_8x8x4_F32F16F16F32_NT>{};
print_latex(mma);
```

![HMMA.8x8x4.NT_Atom.png](images/__________1.png)

The above is equivalent to

```cpp
    TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                                  Layout<Shape<_1,_1,_1>>{},   // Layout of Atoms
                                  Tile<_8,_8,_4>{});           // Tiler
    print_latex(mma);
```

as it is a single atom and has a natural tile size of 8x8x4.

We can create an object akin to a WMMA by using four of these quadpair MMAs:

```cpp
    TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                                  Layout<Shape <_2,_2>,
                                         Stride<_2,_1>>{});   // 2x2 n-major layout of Atoms
    print_latex(mma);
```

![HMMA.8x8x4.NT_2x2.png](images/__________2.png)
This `TiledMMA` replicates the `MMA_Atom` across threads as we can see the `T4` and `T8` and `T12` threads in the `C`-matrix that were not used before. Each quadrant of the `C`-matrix is a replica of the atom’s partitioning pattern for a new quadpair and this replication follows a `(2,2):(2,1)` layout.

The above represents a 16x16x4 MMA now, but we can immediately expand this “tile size” up to 32x32x4 instead:

```cpp
    TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                                  Layout<Shape <_2,_2>,
                                         Stride<_2,_1>>{},  // 2x2 n-major layout of Atoms
                                  Tile<_32,_32,_4>{});      // 32x32x4 tiler
    print_latex(mma);
```

![HMMA.8x8x4.NT_2x2_32x32x4.png](images/__________3.png)
This `TiledMMA` replicates the previous `TiledMMA` across values instead of threads. We can see the `T0V8` and `T16V8` and `T8V8` values in the `C`-matrix that were not used before. Each quadrant of the `C`-matrix is a replica of the previous `TiledMMA`’s partitioning pattern for a new set of values.

Continuing, we see that there are eight values that `T0` receives from the `A`-matrix. Those reads occur at coordinates

```console
T0V0 => ( 0,0)
T0V1 => ( 1,0)
T0V2 => ( 2,0)
T0V3 => ( 3,0)
T0V4 => (16,0)
T0V5 => (17,0)
T0V6 => (18,0)
T0V7 => (19,0)
```

which are separate, but we might prefer them to be next to each other. That is we would like to permute the `M`-mode to create another valid `TiledMMA`.

```cpp
    TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                                  Layout<Shape <_2,_2>,
                                         Stride<_2,_1>>{},       // 2x2 n-major layout of Atoms
                                  Tile<Layout<Shape <_4,_4,_2>,
                                              Stride<_1,_8,_4>>, // Permutation on M, size 32
                                       _32,                      // Permutation on N, size 32 identity
                                       _4>{});                   // Permutation on K, size 4 identity
    print_latex(mma);
```

![HMMA.8x8x4.NT_2x2_32Mx32x4.png](images/__________4.png)

That layout `(4,4,2):(1,8,4)` is read like a scatter permutation, telling the m-coords of the original image where to go in the new image.

```console
old m-coord:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
new m-coord:  0  1  2  3  8  9 10 11 16 17 18 19 24 25 26 27  4  5  6  7 12 13 14 15 20 21 22 23 28 29 30 31
```

This permutes only the M-mode (in `A` and `C` accordingly) and brings the access of all threads to be contiguous in m-coordinates in the `A`-matrix. This is convenient when designing layouts for shared memory or registers, for example. The MMA instructions contained within the image above are now effectively interleaved in the logical m-coordinates. Of course, permutations in the N-mode and K-mode are also valid.

To see how these `TiledMMA`s are used to partition data tensors, see the [`0x_gemm_tutorial.md`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html).
