.. _advanced-data-storage:

Data Storage
============

The general structure of ``Data Storage`` is described in the section :ref:`Terms & Structure: Data Storage <basic-data-storage>`.
This page contains additional information about ``Data Storage``.

Using the Interface
-------------------

When a ``Data Storage`` interface concept is used in a function's argument list, it describes the **minimum** requirements that ``Data Storage`` objects must meet in order to be used to call the function.

.. code-block:: cpp

    void func1(alpaka::concepts::IDataSource auto input){}
    void func2(alpaka::concepts::IView auto input){}

    int main() {
        // fulfil the IDataSource interface, but not the IMdSpan interface
        alpaka::LinearizedIdxGenerator linearizedIdxGenerator;
        // fulfil the IDataSource and the IMdSpan interface, but not the IView interface
        alpaka::MdSpan mdpsan;
        // fulfil the IDataSource, the IMdSpan and the IView interface, but not the IBuffer interface
        alpaka::View view;
        // fulfil the IDataSource, the IMdSpan, the IView and the IBuffer interface
        alpaka::onHost::SharedBuffer sharedBuffer;

        func1(linearizedIdxGenerator);
        func1(mdpsan);
        func1(view);
        func1(sharedBuffer);

        // does not compile, linearizedIdxGenerator does not fulfil the IView interface
        // func2(linearizedIdxGenerator);
        // does not compile, mdspan does not fulfil the IView interface
        // func2(mdpsan);
        func2(view);
        func2(sharedBuffer);
    }

Read-only access via const annotation
-------------------------------------

All objects that implements a ``Data Storage`` interface are const-correct.
If either the value type is ``const`` (e.g., ``alpaka::MdSpan<float const, TExtent, TPitch, TAlignment>``) or the ``Data Storage`` type itself is annotated with ``const`` (e.g., ``alpaka::MdSpan<float, TExtent, TPitch, TAlignment> const``), the values in the storage cannot be changed.

Special case: mutable alpaka::concepts::IDataSource
```````````````````````````````````````````````````

``alpaka::concepts::IDataSource`` only requires that data can be read from a ``Data Storage`` object, but not written to it, regardless of the ``const`` annotation.
However, there are valid cases where a function argument is type-restricted with ``alpaka::concepts::IDataSource`` but requires that data can be written to it, which means that the actual ``Data Storage`` object must implement the ``alpaka::concepts::IMdSpan`` interface.

    .. code-block:: cpp

        namespace alpaka::onAcc {
            struct SimdAlgo{
                // function is provide by alpaka
                ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void concurrent(
                    auto const& acc,
                    auto&& func,
                    alpaka::concepts::IDataSource auto&& data0,
                    alpaka::concepts::IDataSource auto&&... dataN) const
                {
                    // ...
                }
            };
        }

        // user defined
        struct SimdCopyOp {
            constexpr void operator()(auto const&, auto const a, auto c) const
            {
                c = a.load();
            }
        };

        // user defined
        struct Kernel {
            ALPAKA_FN_ACC void operator()(
                auto const& acc,
                auto const& func,
                alpaka::concepts::IDataSource auto const& in,
                alpaka::concepts::MdSpan auto out) const
            {
                auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
                simdGrid.concurrent(acc, in.getExtents(), SimdCopyOp{}, in, out);
            }
        };

``alpaka::onAcc::SimdAlgo.concurrent()`` requires the ``IDataSource`` interface for the ``Data Storage`` object.
The ``IDataSource`` interface only supports reading data. Depending on the user-defined functor, some of the ``Data Storage`` objects must be writable, so they must implement the ``IMdSpan`` interface.
However, the ``IMdSpan`` interface would prevent the user from using generators that only implement the ``IDataSource`` interface.
Therefore, the minimum requirement must be the non-constant ``IDataSource`` interface.

.. _memory-layout-of-multidimensional-data-storage:

Memory Layout of multidimensional Data Storage
----------------------------------------------

There are several functions and parameters for improving the memory layout of multidimensional ``Data Storage`` to enhance application performance.
Alpaka supports ``Pitches``, which optimize memory loads, and ``Alignment``, which is required for vector operations such as AVX on CPUs.
All alpaka functions automatically handle ``Pitches`` and ``Alignment`` during memory access.
However, it is sometimes necessary to process raw memory, for example, when a memory pointer is passed from alpaka to non-alpaka code.
The following section explains how alpaka implements ``Pitches`` and ``Alignment``.

Pitches
```````

A ``Pitch`` consists of ``Extents`` (user data) plus some padding bytes to achieve a specific size in bytes for optimized memory loads.
For example, an Nvidia GPU can load 128 bytes with a single load command.
If we have a matrix with 32-bit integers (4 bytes) and 5 rows with 30 elements each, each row requires 120 bytes.
To ensure that each row can be loaded with a single load command and that no data from a second row is loaded, which would result in some rows requiring multiple load commands, 8 bytes of filler characters are added.

For illustration purposes, we will choose slightly different numbers than in the practical example.
The example has 3 rows with 5 elements each.
Each element has a size of 4 bytes. 2 bytes of padding are added to each row.
Therefore, the size of a row is ``5 elements * 4 Byte/element + 2 Byte = 22 Byte``.

.. figure:: images/2D_padding_example.svg

    Matrix with [3, 5] elements, each element has a size of 4 bytes and 2 bytes of padding per row.


The ``Pitch`` stores the number of bytes required to jump to the next element in a dimension.
In the simplest case, when 1D ``Data Storage`` is used, the size of the value type in bytes is the ``Pitch``.
For example, if the value type is ``float`` (32 bits), the pitch is 4 bytes.
With 2D ``Data Storage``, the first number of a 2D ``Pitch`` stores the size of the row in bytes, i.e., ``number_of_elements * sizeof(value_type) + padding_bytes``.
So if you have a memory pointer and add the ``Pitch[0]``, we jump one column further.
The ``Pitch[1]`` is again the size of the value type.

.. code-block:: cpp

    auto extents = alpaka::Vec{3u, 5u};
    auto buffer = alpaka::onHost::allocUnified<int32_t>(extents);

    extents.x() == 5;
    extents.y() == 3;

    buffer.getPitches().x() == 4; // element size in byte (sizeof(int32_t))
    buffer.getPitches().y() == 22; // 5 elements with 4 bytes + 2 bytes padding

To manually calculate the address of a specific element in a ``Data Storage`` using a given memory pointer of element 0 and the ``Pitch``, use the following code:

.. code-block:: cpp

    // pseudo code
    T* valuePtr = static_cast<T*>(static_cast<std::byte*>(dataPtr) + dot_product(idx, pitches_in_byte))
    // use alpaka 3 functionality
    // idx and pitches are alpaka::Vec
    T* valuePtr = static_cast<T*>(static_cast<std::byte*>(dataPtr) + (idx*pitches).sum())


.. figure:: images/2D_padding_example_linearized.svg

    The linearized memory layout of a matrix with [3, 5] elements, where each element has a size of 4 bytes and 2 bytes of padding per row.

The same applies to higher-dimensional memory. The ``Pitch`` for a dimension describes how many bytes must be added to skip a position in that dimension.
The following example shows 3D memory and the corresponding values for the ``Extent`` and ``Pitch``.

.. figure:: images/3D_padding_example.svg

    Cube with [3, 3, 5] elements, each element has a size of 4 bytes and 2 bytes of padding per row and 22 bytes of padding per side.

.. code-block:: cpp

    auto extents = alpaka::Vec{3u, 3u, 5u};
    auto buffer = alpaka::onHost::allocUnified<int32_t>(extents);

    extents.x() == 5;
    extents.y() == 3;
    extents.z() == 3;

    buffer.getPitches().x() == 4; // element size in byte (sizeof(int32_t))
    buffer.getPitches().y() == 22; // 5 elements with 4 bytes + 2 bytes padding
    buffer.getPitches().z() == 88; // 3 rows with user data and padding, each 22 bytes long + 22 bytes padding row

Alignment
`````````
