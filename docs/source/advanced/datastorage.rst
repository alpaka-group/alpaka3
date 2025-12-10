.. _advanced-data-storage:

Data Storage
============

The general structure of Data Storage is described in the section :ref:`Terms & Structure: Data Storage <basic-data-storage>`.
This page contains additional information about data storage.

Using the Interface
-------------------

When a Data Storage interface concept is used in a function's argument list, it describes the **minimum** requirements that data storage objects must meet in order to be used to call the function.

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

All objects that implements a Data Storage interface are const-correct.
If either the value type is ``const`` (e.g., ``alpaka::MdSpan<float const, TExtent, TPitch, TAlignment>``) or the Data Storage type itself is annotated with ``const`` (e.g., ``alpaka::MdSpan<float, TExtent, TPitch, TAlignment> const``), the values in the storage cannot be changed.

Special case: mutable alpaka::concepts::IDataSource
```````````````````````````````````````````````````

``alpaka::concepts::IDataSource`` only requires that data can be read from a Data Storage object, but not written to it, regardless of the ``const`` annotation.
However, there are valid cases where a function argument is type-restricted with ``alpaka::concepts::IDataSource`` but requires that data can be written to it, which means that the actual Data Storage object must implement the ``alpaka::concepts::IMdSpan`` interface.

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

``alpaka::onAcc::SimdAlgo.concurrent()`` requires the ``IDataSource`` interface for the Data Storage object.
The ``IDataSource`` interface only supports reading data. Depending on the user-defined functor, some of the Data Storage objects must be writable, so they must implement the ``IMdSpan`` interface.
However, the ``IMdSpan`` interface would prevent the user from using generators that only implement the ``IDataSource`` interface.
Therefore, the minimum requirement must be the non-constant ``IDataSource`` interface.

.. _memory-layout-of-multidimensional-data-storage:

Memory Layout of multidimensional Data Storage
----------------------------------------------
