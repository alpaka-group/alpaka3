/* Copyright 2025 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <dlfcn.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>

using namespace alpaka;

// Declare the function type we expect
using alpaka_main_t = int(int argc, char* argv[]);

std::string toLower(std::string str)
{
    std::ranges::transform(str, str.begin(), [](unsigned char c) { return std::tolower(c); });
    return str;
}

int example(int argc, char* argv[], auto const usedApi)
{
    std::filesystem::path p = argv[0];
    auto binaryPath = p.parent_path().string();
    void* handle = nullptr;

    auto prefixList = std::vector<std::string>{".", binaryPath};
    for(auto libPrefix : prefixList)
    {
        auto libraryName = libPrefix + "/libalpaka-ls_" + onHost::getName(usedApi) + ".so";
        handle = dlopen(toLower(libraryName).c_str(), RTLD_LAZY);
        if(handle)
            break;
    }
    if(handle)
    {
        // Lookup function
        auto func = reinterpret_cast<alpaka_main_t*>(
            dlsym(handle, "alpaka_main") // <-- if compiled without extern "C", use mangled
        );

        if(!func)
        {
            std::cerr << "dlsym failed: " << dlerror() << "\n";
            dlclose(handle);
        }

        if(func)
        {
            // Forward argc/argv from main to alpaka_main
            [[maybe_unused]] int ret = func(argc, argv);
            dlclose(handle);
        }
    }

    return EXIT_SUCCESS;
}

void help(char* argv[])
{
    std::cerr << argv[0] << " [-h]" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    int opt;
    while((opt = getopt(argc, argv, "h")) != -1)
    {
        switch(opt)
        {
        case 'h':
            help(argv);
            exit(EXIT_SUCCESS);
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEach([=](auto const& api) { return example(argc, argv, api); }, onHost::apis);
}
