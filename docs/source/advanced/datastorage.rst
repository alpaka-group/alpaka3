.. _advanced-data-storage:

Data Storage
============

The general structure of ``Data Storage`` is described in the section :ref:`Terms & Structure: Data Storage <basic-data-storage>`.
This page contains additional information about ``Data Storage``.

Using the Interface
-------------------

When a ``Data Storage`` interface concept is used in a function's argument list, it describes the **minimum** requirements that ``Data Storage`` objects must meet in order to be used to call the function.

.. literalinclude:: ../../snippets/dataStorage/datastorage_interface.cpp
  :language: cpp
  :start-after: BEGIN-DATASTORAGE-interface
  :end-before: END-DATASTORAGE-interface
  :dedent:

Read-only access via const annotation
-------------------------------------

All objects that implements a ``Data Storage`` interface are const-correct.
If either the value type is ``const`` (e.g., ``alpaka::MdSpan<float const, TExtent, TPitch, TAlignment>``) or the ``Data Storage`` type itself is annotated with ``const`` (e.g., ``alpaka::MdSpan<float, TExtent, TPitch, TAlignment> const``), the values in the storage cannot be changed.

Special case: mutable alpaka::concepts::IDataSource
```````````````````````````````````````````````````

``alpaka::concepts::IDataSource`` only requires that data can be read from a ``Data Storage`` object, but not written to it, regardless of the ``const`` annotation.
However, there are valid cases where a function argument is type-restricted with ``alpaka::concepts::IDataSource`` but requires that data can be written to it, which means that the actual ``Data Storage`` object must implement the ``alpaka::concepts::IMdSpan`` interface.

.. literalinclude:: ../../snippets/dataStorage/datastorage_writeable_datasource.cpp
  :language: cpp
  :start-after: BEGIN-DATASTORAGE-writeableDatasource
  :end-before: END-DATASTORAGE-writeableDatasource
  :dedent:

`alpaka::onAcc::SimdAlgo.concurrent() <https://alpaka3.readthedocs.io/en/latest/doxygen/structalpaka_1_1onAcc_1_1SimdAlgo.html#a5face31ec9941f1eeb69fef9ea9fba01>`_ requires the ``IDataSource`` interface for the ``Data Storage`` object.
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

.. literalinclude:: ../../snippets/dataStorage/datastorage_pitch.cpp
  :language: cpp
  :start-after: BEGIN-DATASTORAGE-pitch2D-example
  :end-before: END-DATASTORAGE-pitch2D-example
  :dedent:

To manually calculate the address of a specific element in a ``Data Storage`` using a given memory pointer of element 0 and the ``Pitch``, use the following code:

.. literalinclude:: ../../snippets/dataStorage/datastorage_pitch.cpp
  :language: cpp
  :start-after: BEGIN-DATASTORAGE-pitch-manual-calculation
  :end-before: END-DATASTORAGE-pitch-manual-calculation
  :dedent:

.. figure:: images/2D_padding_example_linearized.svg

    The linearized memory layout of a matrix with [3, 5] elements, where each element has a size of 4 bytes and 2 bytes of padding per row.

The same applies to higher-dimensional memory. The ``Pitch`` for a dimension describes how many bytes must be added to skip a position in that dimension.
The following example shows 3D memory and the corresponding values for the ``Extent`` and ``Pitch``.

.. figure:: images/3D_padding_example.svg

    Cube with [3, 3, 5] elements, each element has a size of 4 bytes and 2 bytes of padding per row and 22 bytes of padding per side.

.. literalinclude:: ../../snippets/dataStorage/datastorage_pitch.cpp
  :language: cpp
  :start-after: BEGIN-DATASTORAGE-pitch3D-example
  :end-before: END-DATASTORAGE-pitch3D-example
  :dedent:

Alignment
`````````
