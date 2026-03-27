---
title: "Frequently Used Tile Iterator Concepts"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/tile_iterator_concept.html#frequently-used-tile-iterator-concepts"
---

## [Frequently Used Tile Iterator Concepts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#frequently-used-tile-iterator-concepts)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#frequently-used-tile-iterator-concepts "Permalink to this headline")

This section describes several frequently used compositions of the basic tile iterator concepts. They are
listed here as complete type declarations for convenience of the reader.

**_Writeable, Readable, Forward, Contiguous Memory Tile Iterator Concept_.**
This combines several of the basic iterator concepts to
yield a tile iterator capable of loading and storing tiles as well as advancing forward along a traversal sequence.

```c++
/// This tile iterator embodies several of the above:
///
///   - ForwardTileIteratorConcept
///   - ReadableContiguousTileIteratorConcept
///   - WriteableContiguousTileIteratorConcept
///
/// It is restated explicitly for convenience of the reader.
///
struct WriteableReadableForwardContiguousTileIteratorConcept {

  //
  // Data types
  //

  using Element;           ///< Element type composing tile.
  using Shape;             ///< Shape type describing extent of tile. The shape concept depends
                           ///  on iterator implementation
  using Index;             ///< index type used as base for TensorCoord
  using Fragment;          ///< fragment object derived from cutlass::Array<Element, N>

  //
  // Methods
  //

  /// Adds a linear offset in units of Element to internal pointer(s) into tensor
  CUTLASS_DEVICE
  void add_pointer_offset(Index offset);

  /// true if iterators point to same tile, false if otherwise
  CUTLASS_DEVICE bool operator==(WriteableReadableForwardContiguousTileIteratorConcept const &it);

  ///< false if iterators point to same tile, true if otherwise
  CUTLASS_DEVICE bool operator!=(WriteableReadableForwardContiguousTileIteratorConcept const &it);

  /// pre-increment - traverse to next tile in sequence
  CUTLASS_DEVICE
  WriteableReadableForwardContiguousTileIteratorConcept &
  operator++();

  ///< post-increment - traverse to next tile in sequence
  CUTLASS_DEVICE
  WriteableReadableForwardContiguousTileIteratorConcept
  operator++(int);

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag);                    ///< fragment to be loaded from memory

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to be loaded from memory
    Index pointer_offset);                      ///< linear offset (in units of Element) when loading

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag);             ///< fragment to store to memory

  /// Stores a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store to memory
    Index pointer_offset);                      ///< linear offset (in units of Element) when storing
};
```

**_Writeable, Readable, Random Access, Contiguous Memory Tile Iterator Concept_.**
This combines several of the basic iterator concepts to
yield a tile iterator with random access suitable for loading matrix operands for GEMM.

```c++
/// This tile iterator embodies several of the above:
///
///   - ReadableRandomAccessContiguousTileIteratorConcept
///   - WriteableRandomAccessContiguousTileIteratorConcept
///
/// It is restated explicitly for convenience of the reader.
///
struct WriteableReadableRandomAccessContiguousTileIteratorConcept {

  //
  // Data types
  //

  using Element;           ///< Element type composing tile.
  using Shape;             ///< Shape type describing extent of tile. The shape concept depends
                           ///  on iterator implementation
  using Layout;            ///< Layout object mapping
  using TensorRef;         ///< Tensor Reference object
  using TensorCoord;       ///< Logical coordinate in referenced tensor
  using Index;             ///< index type used as base for TensorCoord
  using Fragment;          ///< fragment object derived from cutlass::Array<Element, N>

  //
  // Methods
  //

  /// Adds a linear offset in units of Element to internal pointer(s) into tensor
  CUTLASS_DEVICE
  void add_pointer_offset(Index pointer_offset);

  /// true if iterators point to same tile, false if otherwise
  CUTLASS_DEVICE bool operator==(WriteableReadableRandomAccessContiguousTileIteratorConcept const &it);

  ///< false if iterators point to same tile, true if otherwise
  CUTLASS_DEVICE bool operator!=(WriteableReadableRandomAccessContiguousTileIteratorConcept const &it);

  /// pre-increment - traverse to next tile in sequence
  CUTLASS_DEVICE
  WriteableReadableRandomAccessContiguousTileIteratorConcept &
  operator++();

  ///< post-increment - traverse to next tile in sequence
  CUTLASS_DEVICE
  WriteableReadableRandomAccessContiguousTileIteratorConcept
  operator++(int);

  /// pre-decrement - traverse to previous tile in sequence
  CUTLASS_DEVICE
  WriteableReadableRandomAccessContiguousTileIteratorConcept &
  operator--();

  ///< post-decrement - traverse to previous tile in sequence
  CUTLASS_DEVICE
  WriteableReadableRandomAccessContiguousTileIteratorConcept
  operator--(int);

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  WriteableReadableRandomAccessContiguousTileIteratorConcept & operator+=(TensorCoord const &tile_offset);

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  WriteableReadableRandomAccessContiguousTileIteratorConcept & operator-=(TensorCoord const &tile_offset);

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag);                    ///< fragment to be loaded from memory

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to be loaded from memory
    Index pointer_offset);                      ///< linear offset (in units of Element) when loading

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to be loaded from memory
    TensorCoord const &tile_offset);            ///< loads a tile with a logical offset in units of whole tiles

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to be loaded from memory
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset);                      ///< loads a tile with a logical offset AND a pointer offset

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag);             ///< fragment to store to memory

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store to memory
    Index pointer_offset);                      ///< linear offset (in units of Element) when loading

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void store(
    Fragment const &frag,                       ///< fragment to store to memory
    TensorCoord const &tile_offset);            ///< stores with logical offset in units of whole tiles

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void store(
    Fragment const &frag,                       ///< fragment to store to memory
    TensorCoord const &tile_offset,             ///< stores with logical offset in units of whole tiles
    Index pointer_offset);
};
```
