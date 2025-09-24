Code Changes with Tools
=======================

.. sectionauthor:: Simeon Ehrig

In the alpaka project, we use many tools for developers. One type of these tools are formatting programs for C++ code, CMake code, and more. If a commit contains code changes created by a formatting program, the author of the commit should not be a real person. Instead of a real person, a non-existent ``tool`` author should be used. The reason for this approach is that we can use ``git blame`` to distinguish between what is a functional change in the code and what has only been formatted.

The ``tool`` author can be set with ``git commit --author="Tools <alpaka@hzdr.de>"`` when the commit is created.

If a commit contains code changes and reformatted code, it must be split into two commits. The first commit with the code changes (e.g., changing the formatter configuration) is created with the developer's normal authorship. The second commit contains only the changes made by the tool and has ``tool`` as the authorship.

This PR shows how the changes are split into two commits: https://github.com/alpaka-group/alpaka3/pull/237
