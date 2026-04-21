Formatters and Linters
======================

.. sectionauthor:: Simeon Ehrig

Pre-commit
----------

This project is set up for use with `pre-commit <https://pre-commit.com>`_. Using it will make your code conform with most
of our (easily automatable) code style guidelines automatically. Pre-commit is a tool that manages
`git hooks <https://git-scm.com/docs/githooks>`_ conveniently for you.
In very short (for anything further see `pre-commit <https://pre-commit.com>`_), after running the following in your
working clone of alpaka

.. code-block:: bash

  # if not yet done, install the pre-commit executable following https://pre-commit.com
  cd /path/to/alpaka-working-clone
  pre-commit install

``git`` will run a number of checks prior to every commit and push and will refuse to perform the
pertinent action if they fail. Most of them (like e.g. the formatter) will have automatically altered your working tree
with the necessary changes such that

.. code-block:: bash

  git add -u

will make the next commit pass. Although discouraged, in urgent cases it might be needed to be able to commit even if
the checks fail. For such cases, you can either use

.. code-block:: bash

  git commit --no-verify [...]

to completely skip all checks or use the more fine-grained control described `here <https://pre-commit.com/#temporarily-disabling-hooks>`_.

You can use

.. code-block:: bash

   pre-commit run --all-files

to run all the hooks on all files.

Formatting
----------

This section lists all formatters used in the alpaka project. Normally, you should use pre-commit to format your source code. If you want to format your code without pre-commit, e.g., automatically in your IDE, you will find all the necessary information in the following sections.

To get exactly the same formatted code as expected in CI, you must use the same formatting version locally as in CI. The specific formatting versions can be found in the ``.pre-commit-config.yaml`` file, see :ref:`formatter-versions`.

C++: Clang-Format
`````````````````

For C++ code, we use `clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_ to format the code. Our style guide for C++ code is defined in the ``.clang-format`` file, which is provided in the top-level directory of alpaka.

CMake: Gersemi
``````````````

The CMake code is formatted with `gersemi <https://github.com/BlankSpruce/gersemi>`_.

.. _formatter-versions:

Formatter versions
``````````````````
.. literalinclude:: ../../../.pre-commit-config.yaml
    :language: yaml
    :caption: .pre-commit-config.yaml

Code Changes with Tools
-----------------------

In the alpaka project, we use many tools for developers. One type of these tools are formatting programs for C++ code, CMake code, and more. If a commit contains code changes created by a formatting program, the author of the commit should not be a real person. Instead of a real person, a non-existent ``tool`` author should be used. The reason for this approach is that we can use ``git blame`` to distinguish between what is a functional change in the code and what has only been formatted.

The ``tool`` author can be set with ``git commit --author="Tools <alpaka@hzdr.de>"`` when the commit is created.

If a commit contains code changes and reformatted code, it must be split into two commits. The first commit with the code changes (e.g., changing the formatter configuration) is created with the developer's normal authorship. The second commit contains only the changes made by the tool and has ``tool`` as the authorship.

This PR shows how the changes are split into two commits: https://github.com/alpaka-group/alpaka3/pull/237
