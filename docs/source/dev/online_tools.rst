Online Tools
============

This section introduces some C++ online tools that are helpful in developing alpaka.

Compiler Explorer (Godbolt)
---------------------------

`Compiler Explorer <https://godbolt.org>`__ is a website where you can compile C++ (and other code) with different compilers, view the assembly code, and execute the code.

alpaka provides a header file that can be included into Compiler Explorer. This allows you to easily prototype alpaka ideas with different compilers in your web browser. You can find a link to an alpaka Compiler Explorer example in the :ref:`code example section <code-example>`.

C++ Insights
------------

`C++ Insights <https://cppinsights.io/>`__ is a tool that shows how a compiler implements/resolves C++ functions. To do this, C++ is converted into modified C++. For example, it shows functions such as template resolution or how a range-based loop is implemented. C++ Insights does not support external header files. Therefore, alpaka cannot be integrated.

C++ Insights is very well suited for prototyping and understanding how C++ code works, especially templates.

C++ Standard references
-----------------------

- Web search for keywords in the C++ standard: 
    - https://wg21.cmeerw.net/cppdraft/search
    - https://search.cpp-lang.org/
- Overview of all terms in the C++ standard: https://eel.is/c++draft/
- HTML and PDF versions of the C++ standards: https://github.com/timsong-cpp/cppwp
- Drafts and papers of the C++ standard features: https://wg21.link/
- What has changed between the C++ standard versions?: https://cppevo.dev/
